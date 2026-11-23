//===- EmbFinalizeReductions.cpp - Finalize reductions in device code -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Finalize reductions in embedded modules. The finalization is only needed
// when a warp-shuffle technique is used to compute the contribution of each
// warp/block before accumulating that result into the final reduction. In this
// case, we must ensure that the call to the warp shuffle intrinsic is executed
// from every thread because the current implementation of the intrinsic assumes
// that all threads in the warp participate in the shuffle. The pseudocode below
// shows what the kernel corresponding to a tapir reduction loop would look like
// after some function simplification passes have been run on the spawned loop.
//
//     if (idx_x < tc_x && ...) {
//       local1 = <unit1>;
//       local2 = <unit2>;
//       ...
//       kit.reduce(&local1, ReductionStrategy::Naive, ...)
//       kit.reduce(&local2, ReductionStrategy::Naive, ...)
//       kit.reduce(&res1, ReductionStrategy::GPUWarpShuffle, local1, ...);
//       kit.reduce(&res2, ReductionStrategy::GPUWarpShuffle, local2, ...);
//     }
//
// Here, local is a variable into which a per-thread local reduction is
// computed in the first call to kit.reduce. This local value is then reduced
// into the final result using the GPUWarpShuffle strategy. idx_x is the
// equivalent of the induction variable of the outermost tapir loop in the nest
// that was lowered to for this GPU kernel, while tc_x is the trip count of that
// loop. The conditional contains similar checks for the induction variables and
// trip counts of the other tapir loops in the nest.
//
// For strategies that utilize warp shuffles, this will result in incorrect
// results at runtime since the GPU's execution model requires all threads in a
// warp to participate in a warp shuffle. This pass transforms the code above
// to the following:
//
//     local1 = <unit1>;
//     local2 = <unit2>;
//     if (idx_x < tc_x && ...) {
//       ...
//       kit.reduce(&local1, ReductionStrategy::Naive, ...)
//       kit.reduce(&local2, ReductionStrategy::Naive, ...)
//     }
//    __syncwarp();
//    kit.reduce(&res1, ReductionStrategy::GPUWarpShuffle, local1, ...);
//    __syncwarp();
//    kit.reduce(&res2, ReductionStrategy::GPUWarpShuffle, local2, ...);
//
// This ensures that every thread in a warp performs a reduction. This
// transformation is only valid if all of the following are true:
//
//   - The only uses of the destination of the reduction are its own
//     initialization and the reduce instruction itself
//
//   - The kernel function has a single return instruction that post-dominates
//     the entry block.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/EmbFinalizeReductions.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/FuncAttrs.h"
#include "kitsune/Core/FuncUtils.h"
#include "kitsune/Core/IntrinsicUtils.h"
#include "kitsune/Core/Reductions.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"

#define DEBUG_TYPE "emb-finalize-reductions"

using namespace llvm;

namespace {

class EmbFinalizeReductions {
protected:
  LoopInfo &li;
  PostDominatorTree &pdt;

protected:
  Reductions getReductions(Function &f);
  AllocaInst *getLocal(const ReductionInfo &redxn);
  bool check(Function &f, const Reductions &redxns);

public:
  EmbFinalizeReductions(LoopInfo &li, PostDominatorTree &pdt)
      : li(li), pdt(pdt) {}

  bool runOn(Function &f, TTID tt);
};

} // namespace

static bool isWarpShuffle(ReductionStrategy strategy) {
  return strategy == ReductionStrategy::GPUWarpShuffle ||
         strategy == ReductionStrategy::GPUWarpShuffleWithSharedMemory;
}

Reductions EmbFinalizeReductions::getReductions(Function &f) {
  SmallVector<ReductionInfo, 0> redxns;
  for (ReductionInfo redxn : ::getReductions(f))
    if (isWarpShuffle(redxn.strategy))
      redxns.push_back(redxn);
  return redxns;
}

AllocaInst *EmbFinalizeReductions::getLocal(const ReductionInfo &redxn) {
  assert(isWarpShuffle(redxn.strategy) &&
         "Reduction strategy is a warp shuffle");

  // When the reduction strategy is a warp shuffle, the value being reduced will
  // be the local reduction that was computed. This should have been obtained
  // by loading from the alloca into which the local reduction is computed.
  if (auto *load = dyn_cast<LoadInst>(redxn.getValue()))
    if (auto *alloca = dyn_cast<AllocaInst>(load->getPointerOperand()))
      return alloca;
  return nullptr;
}

bool EmbFinalizeReductions::check(Function &f, const Reductions &redxns) {
  assert(!redxns.empty() && "At least one reduction found");

  if (!hasKernelAttr(f))
    return emitDiagnostic(f, DiagID::ErrGeneric,
                          "reductions with the warp shuffle strategy are only "
                          "allowed in kernel functions");

  for (const ReductionInfo &redxn : redxns) {
    Value *call = redxn.call;
    Value *dest = redxn.getDest();

    if (!getLocal(redxn))
      return emitDiagnostic(
          *call, DiagID::ErrGeneric,
          "value of final reduction  must be loaded from an alloca");

    unsigned numUses = dest->getNumUses();
    if (numUses != 1)
      return emitDiagnostic(*dest, DiagID::ErrNumUses, 1, numUses);
  }

  ReturnInst *ret = getUniqueInstIn<ReturnInst>(f);
  if (!ret)
    return emitDiagnostic(f, DiagID::ErrFuncNoUniqueReturn);

  BasicBlock *bbRet = ret->getParent();
  BasicBlock *bbEntry = &f.getEntryBlock();
  if (!pdt.dominates(bbRet, bbEntry))
    return emitDiagnostic(f, DiagID::ErrFuncExitNotPostDomEntry);

  return true;
}

bool EmbFinalizeReductions::runOn(Function &f, TTID tt) {
  Reductions redxns = this->getReductions(f);
  if (redxns.empty())
    return false;
  else if (!check(f, redxns))
    exitOnError();

  LLVMContext &ctx = f.getContext();
  Value *ctt = toConstant(tt, ctx);

  BasicBlock::iterator initPt = f.getEntryBlock().getFirstNonPHIOrDbgOrAlloca();
  ReturnInst *ret = getUniqueInstIn<ReturnInst>(f);
  for (const ReductionInfo &redxn : redxns) {
    CallInst *call = redxn.call;
    Type *type = redxn.getType();
    Value *unit = redxn.getUnit();
    AllocaInst *local = getLocal(redxn);

    // Initialize the local reduction destination in the function entry block
    // since the final reduction must be performed unconditionally.
    new StoreInst(unit, local, initPt);

    // Move the final reduction call to just before the return instruction.
    call->moveBefore(ret->getIterator());

    // It is not clear that we need to synchronize the threads in the warp here,
    // but it is safe. We should try to figure out if it is always safe.
    createCall(call->getIterator(), Intrinsic::kit_gpu_warp_sync, {ctt});

    // The value being reduced will be defined conditionally. Instead, just
    // load the value from the local reduction here. We could insert a PHI node,
    // but that is more complicated.
    Value *v = new LoadInst(type, local, "", call->getIterator());
    call->setArgOperand(ReductionInfo::argNumValue, v);
  }

  return true;
}

bool EmbFinalizeReductionsPass::run(TTID tt, Function &devF,
                                    FunctionAnalysisManager &devAM,
                                    Module &hostM,
                                    ModuleAnalysisManager &hostAM) {
  LoopInfo &li = devAM.getResult<LoopAnalysis>(devF);
  PostDominatorTree &pdt = devAM.getResult<PostDominatorTreeAnalysis>(devF);

  return EmbFinalizeReductions(li, pdt).runOn(devF, tt);
}

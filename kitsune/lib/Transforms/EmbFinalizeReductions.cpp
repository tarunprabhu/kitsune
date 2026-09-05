//===- EmbFinalizeReductions.cpp - Finalize reduction kernels -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Finalize GPU kernels that perform a reduction. These kernels will have been
// obtained from tapir loops that contain reductions.
//
// These kernels are expected to be of the form:
//
//     void f(long beg, long end, ..., void *globalResult, long n) {
//       for (long i = beg; i < end; ++i) {
//          ...
//          v = ...;
//          @llvm.kit.reduce.0(globalResult, REDUCE_OP, .., v, unit, ...);
//       }
//     }
//
// Here, globalResult is memory allocated with directly in GPU global memory, or
// in UVM into which the reduction is performed. This pass will transforms this
// to the following:
//
//     void f(long beg, long end, ..., void *globalResult, long n) {
//       long localResult = unit;
//       for (long i = beg; i < end; ++i) {
//          ...
//          v = ...;
//          @llvm.kit.reduce.0(localResult, REDUCE_OP, v);
//       }
//       cooperativeReduce(globalResult, REDUCE_OP, localResult);
//     }
//
// Here, the reduction is performed in a thread-local variable. This computed
// value is then reduced into the globalResult using `cooperativeReduce`. This
// is a function that indicates that the threads on the GPU will cooperate to
// carry out a reduction. The most naive implementation of this would be where
// every thread performs an atomic reduction into globalResult. Most
// sophisticated techniques such as warp shuffles may also be employed.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/EmbFinalizeReductions.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/FuncAttrs.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/Reductions.h"
#include "kitsune/Support/CommandLineOptions.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"

#define DEBUG_TYPE "emb-finalize-reductions"

using namespace llvm;

namespace {

// The various reduction strategies that are available. This is the strategy
// used to perform the final reduction into the result.
enum Strategy {
  // Each thread accumulates its contribution directly into the result variable
  // using an atomic operation. This is the most naive approach possible, and is
  // unlikely to ever be useful. It may be useful when comparing approaches, so
  // it is retained.
  Direct = 1,

  // Each thread in a block writes its contribution to the result into a
  // dedicated location in shared memory. These are then reduced into a single
  // value for the block. This is accumulated into the result variable using an
  // atomic operation.
  SharedMemory,

  // Each thread in a warp computes its contribution to the result, which are
  // the reduced to a single value for the warp using warp shuffles. A single
  // thread in the warp will accumulate this directly into the result variable
  // using an atomic operation.
  WarpShuffle,

  // Each thread in a warp computes its contribution to the result, which are
  // the reduced to a single value for the warp using warp shuffles. A single
  // thread in the warp writes this to a dedicated location in shared memory.
  // These are then reduced to a single value for the block. This value is
  // accumulated into the result variable using an atomic operation.
  WarpShuffleWithSharedMemory,
};

} // namespace

// The default strategy to use when reducing into the final result.
static Strategy defaultStrategy = Strategy::WarpShuffle;

static cl::opt<Strategy> clStrategy(
    "tapir-gpu-reduce-mode",
    cl::desc("The strategy to use for tapir reduction loops on GPUs"),
    cl::init(defaultStrategy), cl::value_desc("strategy"),
    cl::cat(cl::catKitClOpts),
    cl::values(clEnumValN(Strategy::Direct, "direct", ""),
               clEnumValN(Strategy::SharedMemory, "mem", ""),
               clEnumValN(Strategy::WarpShuffle, "wshf", ""),
               clEnumValN(Strategy::WarpShuffleWithSharedMemory, "wshfmem",
                          "")));

template <typename T, typename... Args>
static bool complain(const T &irElem, DiagID diag, Args &&...args) {
  emitDiagnostic(irElem, diag, args...);
  exitOnError();
}

// Perform some analysis on the loop given the reductions that are performed in
// it and determine which reduction strategy to use.
static Strategy chooseStrategy(const Loop &loop,
                               const SmallVectorImpl<ReductionInfo> &redxns) {
  if (clStrategy.getNumOccurrences())
    return clStrategy;

  // TODO: Actually perform an analysis instead of just returning a default.
  return defaultStrategy;
}

// Allocate a stack variable into which the per-thread reduction will be
// stored. The reductions may not be as simple as `res += i`, or even
// `res &= a[i]`. A single thread may contain a loop that reduces multiple times
// into the result variable `res`. For instance, something like this:
//
//     parallel_for (int i = 0; i < n; ++i) {
//       for (auto j : x)
//         for (auto k : y)
//           res += f(i, j, k);
//     }
//
// In such cases, it is better to create a local variable to accumulate all the
// values of `f` that are computed in a single thread, then accumulate the final
// result into `res`. Note that the initial value of res here is set to 0
// because that is the identity for the += operator.
//
//     parallel_for (int i = 0; i < n; ++i) {
//       decltype(res) local = 0;
//       for (auto j : x)
//         for (auto k : y)
//           local += f(i, j, k);
//       res += local;
//     }
//
// This transformation is absolutely necessary for an implementation based on
// warp shuffles since all threads in the warp must perform the shuffle. If,
// for whatever reason, we were to forgo this implementation, and reduce into
// `res` directly, this would also reduce the number of atomic reduce operations
// that would be performed on `res`.
static Value *allocLocalResult(Function &f, const ReductionInfo &redxn) {
  LLVM_DEBUG(dbgs() << "FinalizeGPUReduction: Allocate per-thread result\n");

  Type *type = redxn.getType();
  Value *unit = redxn.getUnit();
  unsigned size = redxn.elemSize;

  BasicBlock &entry = f.getEntryBlock();
  IRBuilder<> builder(&*entry.getFirstNonPHIOrDbgOrAlloca());

  Type *localType = redxn.getResultBufferType();
  AllocaInst *local = builder.CreateAlloca(localType, nullptr, "reduc.local");
  if (isa<PointerType>(type))
    builder.CreateMemCpy(local, MaybeAlign(), unit, MaybeAlign(), size);
  else
    builder.CreateStore(unit, local);

  return local;
}

// Reduce into the local result variable \p local. \p local is the value
// returned by \ref allocLocalResult.
static void reduceIntoLocal(Value *local, const ReductionInfo &redxn) {
  LLVM_DEBUG(dbgs() << "FinalizeGPUReduction: Reduce into local result\n");

  // The only allowed use of the destination of a reduction is in the reduction
  // intrinsic call. Therefore, all we need to do to reduce into the local
  // instead is to replace the destination operand in the reduction call.
  LLVMContext &ctx = local->getContext();
  PointerType *ptr = PointerType::getUnqual(ctx);

  // The local variable may have to be cast because the local variable may be in
  // a non-default address space on some architectures.
  Value *cst = local;
  if (local->getType() != ptr) {
    InsertPosition insertPt = redxn.call->getIterator();
    cst = CastInst::CreatePointerCast(local, ptr, "", insertPt);
  }
  redxn.call->setArgOperand(2, cst);
}

// Perform the reduction into the final result. \p local is the value being
// reduced and is the contribution of the thread to the final reduction.
// \p strategy is the strategy to use to perform the reduction.
static void reduceIntoFinal(Loop &loop, Strategy strategy, Value *local,
                            Value *dest, const ReductionInfo &redxn) {
  auto sanityCheck = [](const Loop &loop) {
    assert(getUniqueNonDeadEndExitBlock(loop) &&
           "Main kernel loop must have a unique non-deadend exit block");
  };

  LLVM_DEBUG(dbgs() << "FinalizeGPUReduction: Reduce into final result\n");
  sanityCheck(loop);

  Value *tt = redxn.getTTV();
  Value *op = redxn.getReduceOpV();
  Type *type = redxn.getType();
  Value *size = redxn.getElemSizeV();
  Value *unit = redxn.getUnit();
  Value *reducer = redxn.getReducer();

  BasicBlock *exit = getUniqueNonDeadEndExitBlock(loop);
  IRBuilder<> builder(&*exit->begin());
  Value *value = builder.CreateLoad(type, local);

  SmallVector<Type *, 2> overloadTys = redxn.getOverloadTypes();
  SmallVector<Value *, 8> args = {tt, op, dest, size, value, unit, reducer};
  args.append(redxn.getExtraArgs());

  auto getIntrinsic = [](Strategy strategy) -> Intrinsic::ID {
    switch (strategy) {
    case Strategy::Direct: return Intrinsic::kit_gpu_reduce_direct;
    case Strategy::SharedMemory: return Intrinsic::kit_gpu_reduce_shared_memory;
    case Strategy::WarpShuffle: return Intrinsic::kit_gpu_reduce_warp_shuffle;
    case Strategy::WarpShuffleWithSharedMemory:
      return Intrinsic::kit_gpu_reduce_warp_shuffle_shared_memory;
    }
    llvm_unreachable("reduceIntoShadowMem: Strategy not handled");
  };
  builder.CreateIntrinsic(getIntrinsic(strategy), overloadTys, args);
}

static bool check(Function &f, LoopInfo &li) {
  // This pass is only run on outlined kernel functions. The function is
  // expected to have a single loop. The exit block of the loop must contain a
  // single return instruction.
  if (li.getTopLevelLoops().size() != 1)
    return complain(f, DiagID::ErrKernelNoMainLoop);

  Loop &loop = *li.getTopLevelLoops()[0];
  if (!getUniqueNonDeadEndExitBlock(loop))
    return complain(loop, DiagID::ErrKernelMainLoopNoExitBlock);

  return true;
}

static bool run(Function &f, LoopInfo &li) {
  if (!check(f, li))
    exitOnError();

  Loop &loop = *li.getTopLevelLoops()[0];
  if (!hasReductionAttr(loop))
    return false;

  SmallVector<ReductionInfo, 1> redxns = collectReductions(loop);
  Strategy strategy = chooseStrategy(loop, redxns);
  for (const ReductionInfo &redxn : redxns) {
    // We have to save the destination because the destination in the reduce
    // call will change during the course of this transformation.
    Value *dest = redxn.getDest();
    Value *local = allocLocalResult(f, redxn);
    reduceIntoLocal(local, redxn);
    reduceIntoFinal(loop, strategy, local, dest, redxn);
  }

  return redxns.size();
}

bool EmbFinalizeReductionsPass::run(TTID tt, Module &devM,
                                    ModuleAnalysisManager &devAM, Module &hostM,
                                    ModuleAnalysisManager &hostAM) {
  FunctionAnalysisManager &fam =
      devAM.getResult<FunctionAnalysisManagerModuleProxy>(devM).getManager();

  bool changed = false;
  for (Function &f : devM) {
    if (hasKernelAttr(f)) {
      LoopInfo &li = fam.getResult<LoopAnalysis>(f);
      changed |= ::run(f, li);
    }
  }

  return changed;
}

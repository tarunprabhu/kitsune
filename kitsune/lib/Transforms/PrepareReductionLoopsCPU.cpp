//===- PrepareReductionLoopsCPU.cpp - Transform reduction loops for CPU ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transform tapir loops that perform reductions to a form that is suitable for
// parallel execution.
//
//     int32_t r_sum = 0;
//     int64_t r_and = 0;
//     parallel_for (int i = 0; i < n; ++i) {
//         kit.reduce(&r_sum, sizeof(r_sum), i, 0, &sum);
//         kit.reduce(&r_and, sizeof(r_and), i, 1, &and);
//     }
//
// This pass will transform this into the following for parallel execution on a
// CPU.
//
//     int64_t numThreads = kit.num.cpu.threads();
//     int64_t itersPerThread = (n + numThreads - 1) / numThreads;
//     parallel_for (int j = 0; j < numPartials; ++j) {
//         int32_t l_sum = 0;
//         int64_t l_and = 1;
//         int start = j * itersPerThread;
//         int end = std::min(start + itersPerThread, n);
//         for (int i = start; i < end; ++i) {
//             kit.reduce(KIT_ADD, &l_sum, sizeof(r_sum), i, 0, &sum);
//             kit.reduce(KIT_AND, &l_and, sizeof(r_and), i, 1, &and);
//         }
//         atomicReduce(&r_sum, l_sum, KIT_ADD);
//         atomicReduce(&r_and, l_and, KIT_AND);
//     }
//
// Here, an outer parallel loop performs a partial reduction on each CPU thread.
// Each iteration of the parallel loop will perform a sequential reduction into
// a thread-private variable. Once this has been computed, the local reduction
// is accumulated into the final destination using an atomic operation.
//
//===----------------------------------------------------------------------===//

#include "LoopWrapping.h"
#include "LowerReduceIntrinsics.h"
#include "PrepareReductionLoops.h"
#include "PrepareTapirLoopsImpl.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/ModuleUtils.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"

#define DEBUG_TYPE "kit-prepare"

using namespace llvm;

namespace {

class PrepareReductionLoopCPU : public detail::PrepareTapirLoopCPU {
private:
  Value *allocPartial(Loop &innerLoop, const ReductionInfo &redxn);
  void reduceIntoPartial(Loop &loop, Value *partial,
                         const ReductionInfo &redxn);
  void reducePartialIntoFinal(Loop &loop, Value *partial, Value *dest,
                              const ReductionInfo &redxn);
  void updateReductionStrategy(CallInst *call);

public:
  PrepareReductionLoopCPU(DominatorTree &dt, LoopInfo &li, MemorySSA &mssa,
                          ScalarEvolution &se, TaskInfo &ti)
      : detail::PrepareTapirLoopCPU(dt, li, mssa, se, ti) {}

  bool run(TapirLoopInfo &tapirLoop,
           const SmallVectorImpl<ReductionInfo> &reductions);
};

} // namespace

// Allocate a buffer into which a partial reduction will be accumulated. The
// buffer will be allocated in the inner loop preheader. This will also
// initialize the allocated buffer with the unit value for the reduction. Since
// the body of the outer loop will eventually be outlined ny the loop-spawning
// pass, this will ensure that the alloca eventually ends up in a function, not
// within a loop.
Value *PrepareReductionLoopCPU::allocPartial(Loop &innerLoop,
                                             const ReductionInfo &redxn) {
  auto sanityCheck = [](const Loop &innerLoop) -> void {
    assert(innerLoop.getLoopPreheader() && "Inner loop must have a preheader");
  };

  LLVM_DEBUG(dbgs() << "PrepareReductionCPU: Allocate buffer for partial\n");
  sanityCheck(innerLoop);

  Type *type = redxn.getType();
  Value *unit = redxn.getUnit();
  unsigned size = redxn.elemSize;

  BasicBlock &bb = *innerLoop.getLoopPreheader();
  IRBuilder<> builder(bb.getTerminator());

  // The alloca is added to the body of the outer loop because we need it to be
  // "local" to the parallel loop. Once the loop-spawning pass has outlined the
  // body into separate function, the kit-hoist-allocas pass will be run which
  // will hoist it to the entry block of that function.
  Type *partialTy = redxn.getResultBufferType();
  AllocaInst *partial =
      builder.CreateAlloca(partialTy, nullptr, "reduc.partial");
  if (isa<PointerType>(type))
    builder.CreateMemCpy(partial, MaybeAlign(), unit, MaybeAlign(), size);
  else
    builder.CreateStore(unit, partial);

  return partial;
}

// Change the destination of the reduction to accumulate into the buffer
// allocated for the partial reduction. Essentially, this changes the
// destination argument of the reduction intrinsic call.
void PrepareReductionLoopCPU::reduceIntoPartial(Loop &innerLoop, Value *partial,
                                                const ReductionInfo &redxn) {
  LLVM_DEBUG(dbgs() << "PrepareReductionCPU: Reduce into partial\n");

  IRBuilder<> builder(redxn.call);
  Value *cst = builder.CreateAddrSpaceCast(partial, redxn.getDest()->getType());
  redxn.call->setArgOperand(ReductionInfo::argNumDest, cst);
}

// Once the partial reduction has been computed, reduce this into the final
// destination of the reduction. This is done using LLVM's AtomicRMW instruction
// when it supports the reduction operator, or a custom atomic operation if it
// does not.
void PrepareReductionLoopCPU::reducePartialIntoFinal(
    Loop &innerLoop, Value *partial, Value *dest, const ReductionInfo &redxn) {
  auto sanityCheck = [](const Loop &innerLoop) -> void {
    assert(getExitBlockFromLatch(innerLoop) &&
           "Loop must have a unique exit block");
  };

  LLVM_DEBUG(dbgs() << "PrepareReductionCPU: Allocate buffer for partial\n");
  sanityCheck(innerLoop);

  Type *elemType = redxn.getType();
  std::optional<AtomicRMWInst::BinOp> atomicOp = getAtomicOp(redxn.reduceOp);
  if (!atomicOp)
    emitDiagnostic(innerLoop, DiagID::ErrNYI,
                   "Reduction operator not supported by AtomicRMWInst");

  Align align = getPointerAlignment(*getModule(innerLoop));
  BasicBlock &bb = *getExitBlockFromLatch(innerLoop);
  IRBuilder<> builder(bb.getTerminator());

  Value *v = builder.CreateLoad(elemType, partial);
  builder.CreateAtomicRMW(*atomicOp, dest, v, align, AtomicOrdering::Monotonic);
}

// Update the strategy to use to perform this reduction.
void PrepareReductionLoopCPU::updateReductionStrategy(CallInst *call) {
  detail::setStrategy(call, ReductionStrategy::Naive);
}

// Transform a tapir reduction loop. Add the tapir.loop.prepared attribute to
// the loop and return true if anything other than this attribute was changed.
// For example, if the loop does not perform any actual reductions, the
// attribute will be added to the loop, but no other transformations will be
// performed. In such cases, simply return false.
bool PrepareReductionLoopCPU::run(
    TapirLoopInfo &tapirLoop,
    const SmallVectorImpl<ReductionInfo> &reductions) {
  Loop &loop = *tapirLoop.getLoop();
  LLVM_DEBUG(dbgs() << "PrepareReductionCPU: BEGIN '" << getName(loop)
                    << "'\n");

  // Calling the base class implementation will block the loop, so we will have
  // an outer parallel loop and inner serial loop.
  detail::PrepareTapirLoopCPU::run(tapirLoop);

  // This inserts code for the allocation, initialization, and use of the
  // partial reduction buffers.
  for (const ReductionInfo &redxn : reductions) {
    // We have to save the destination because the operands to the original call
    // will be changed.
    Value *dest = redxn.getDest();
    Value *partial = allocPartial(loop, redxn);
    reduceIntoPartial(loop, partial, redxn);
    reducePartialIntoFinal(loop, partial, dest, redxn);
    updateReductionStrategy(redxn.call);
  }

  return true;
}

bool llvm::detail::prepareReductionLoopForCPU(
    TapirLoopInfo &tapirLoop, const SmallVectorImpl<ReductionInfo> &reductions,
    DominatorTree &dt, LoopInfo &li, MemorySSA &mssa, ScalarEvolution &se,
    TaskInfo &ti) {
  return PrepareReductionLoopCPU(dt, li, mssa, se, ti)
      .run(tapirLoop, reductions);
}

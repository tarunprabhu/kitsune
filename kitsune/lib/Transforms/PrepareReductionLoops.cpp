//===- PrepareReductionLoops.cpp - Transform tapir reduction loops --------===//
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
// Consider the loop shown below
//
//     int32_t r_sum = 0;
//     int64_t r_and = 1;
//     parallel_for (int i = 0; i < n; ++i) {
//         r_sum += i;
//         r_and &= 1;
//     }
//
// Frontends are expected to use the kit.reduce.0 intrinsic to represent the
// loop above
//
//     void sum(int32_t* res, int32_t v) {
//         *res += v;
//     }
//
//     void and(int64_t* res, int64_t v) {
//         *res &= v;
//     }
//
//     parallel_for (int i = 0; i < n; ++i) {
//         kit.reduce.0(&r_sum, sizeof(r_sum), i, 0, &sum);
//         kit.reduce.0(&r_and, sizeof(r_and), i, 1, &and);
//     }
//
// This pass will transform this into the following for parallel execution on a
// CPU.
//
//     int64_t numPartials = ...
//     parallel_for (int j = 0; j < numPartials; ++j) {
//         int start = j * sizePartial;
//         int end = std::min(start + sizePartial, n);
//         int32_t *local1 = (int32_t*)malloc(sizeof(int32_t));
//         int64_t *local2 = (int64_t*)malloc(sizeof(int64_t));
//         *local1 = 0;
//         *local2 = 1;
//         for (int i = start; i < end; ++i) {
//             kit.reduce.0(local1, sizeof(r_sum), i, 0, &sum);
//             kit.reduce.0(local2, sizeof(r_and), i, 1, &and);
//         }
//         atomicReduce(&sum, &r_sum, *local1);
//         atomicReduce(&and, &r_and, *local2);
//         free(local1);
//         free(local2);
//     }
//
// Here, numPartials is the number of partial reductions that are to be
// performed in parallel. An outer parallel loop is added to carry these out.
// Each iteration of the parallel loop will perform a sequential reduction. This
// is followed by calls to kit.reduce.1 which computes the final result from the
// partial reductions.
//
// The code below is the transformation that is carried out for GPU's.
// FIXME: This is obviously not correct since we cannot malloc on the GPU.
// Clearly, this means that reductions will not work on the GPU at this time.
//
//     int64_t numPartials = ...;
//     parallel_for (int j = 0; j < numPartials; ++j) {
//         int32_t *local1 = (int32_t*)malloc(sizeof(int32_t));
//         int64_t *local2 = (int64_t*)malloc(sizeof(int64_t));
//         *local1 = 0;
//         *local2 = 1;
//         for (int i = j; i < n; i += numPartials) {
//             kit.reduce.0(local1, sizeof(r_sum), i, 0, &sum);
//             kit.reduce.0(local2, sizeof(r_and), i, 1, &and);
//         }
//         atomicReduce(&sum, &r_sum, *local1);
//         atomicReduce(&and, &r_and, *local2);
//         free(local1);
//         free(local2);
//     }
//
// Note that this pass will not lower any existing reduce intrinsics - in fact,
// it introduces additional calls to Kitsune's reduce intrinsics. These calls
// will be lowered in a different pass.
//
// The transformations shown above are "generic" approaches that this pass will
// use when lowering for CPU and GPU respectively. Depending on the tapir target
// set on the loop, a different transformation may be used.
//
//===----------------------------------------------------------------------===//

#include "PrepareReductionLoops.h"
#include "LoopWrapping.h"
#include "kitsune/Core/AddrSpace.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/InstUtils.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/Reductions.h"
#include "kitsune/Core/TTUtils.h"
#include "kitsune/Core/TapirLoopUtils.h"
#include "kitsune/Core/ValueUtils.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/PriorityWorklist.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

#define DEBUG_TYPE "kit-prepare"

using namespace llvm;

namespace {

/// Information about a single reduction in the tapir reduction loop. A loop
/// may perform more than one such reduction.
struct ReductionInfo {
  CallBase *call = nullptr;      ///< Call to the kit_reduce_0 intrinsic
  Value *tt = nullptr;           ///< The TTID of the tapir reduction loop
  ReduceOp reduceOp;             ///< The reduction operator
  Value *dest = nullptr;         ///< The destination for the reduced value
  Value *elemSize = nullptr;     ///< The size (in bytes) of the reduced result
  Value *unit = nullptr;         ///< The unit value for the reduction
  Value *reducer = nullptr;      ///< The reducer function
  SmallVector<Value *, 1> extra; ///< Extra arguments for the reducer function

  ReductionInfo(CallBase *call) : call(call) {
    tt = call->getArgOperand(0);
    reduceOp = *fromConstant<ReduceOp>(*cast<Constant>(call->getArgOperand(1)));
    dest = call->getArgOperand(2);
    elemSize = call->getArgOperand(3);
    unit = call->getArgOperand(5);
    reducer = call->getArgOperand(6);
    for (unsigned i = 7; i < call->arg_size(); ++i)
      extra.push_back(call->getArgOperand(i));
  }
};

/// Base class to transform tapir reduction loops for parallel execution.
class PrepareReductionLoop {
private:
  DominatorTree &dt;
  LoopInfo &li;
  MemorySSA &mssa;
  ScalarEvolution &se;
  TaskInfo &ti;

  DomTreeUpdater dtu;
  MemorySSAUpdater mssau;

private:
  SmallVector<ReductionInfo, 1> collectReductions(Loop &loop);
  Value *computeNumPartialReductions(Loop &outerLoop, const Loop &innerLoop);
  Value *allocPartial(Loop &innerLoop, const ReductionInfo &info);
  void reduceIntoPartial(Loop &loop, Value *partial, const ReductionInfo &info);
  void reducePartialIntoFinal(Loop &loop, Value *partial,
                              const ReductionInfo &info);
  void freePartial(Loop &innerLoop, Value *partial);
  void updateOuterLoopIV(Loop &outerLoop, Value *numPartials);
  void updateInnerLoopIVCPU(Loop &outerLoop, Loop &innerLoop,
                            Value *numPartials);
  void updateInnerLoopIVGPU(Loop &outerLoop, Loop &innerLoop,
                            Value *numPartials);
  void updateInnerLoopGuardCondition(Loop &innerLoop, Value *numPartials);
  void parallelizeOuterLoop(Loop &outerLoop, Loop &loop);
  void serializeInnerLoop(Loop &loop);

public:
  PrepareReductionLoop(DominatorTree &dt, LoopInfo &li, MemorySSA &mssa,
                       ScalarEvolution &se, TaskInfo &ti)
      : dt(dt), li(li), mssa(mssa), se(se), ti(ti),
        dtu(dt, DomTreeUpdater::UpdateStrategy::Eager), mssau(&mssa) {}

  bool run(TapirLoopInfo &tapirLoop);
};

} // namespace

// Collect the reduction intrinsics called in the loop being transformed,
// \p loop.
SmallVector<ReductionInfo, 1>
PrepareReductionLoop::collectReductions(Loop &loop) {
  SmallVector<ReductionInfo, 1> reductions;

  for (BasicBlock *bb : loop.getBlocks())
    for (Instruction &inst : *bb)
      if (auto *call = dyn_cast<CallBase>(&inst))
        if (call->getIntrinsicID() == Intrinsic::kit_reduce_0)
          reductions.emplace_back(call);

  return reductions;
}

// Insert code to calculate the number of partial reductions to use. On CPU's,
// this inserts a call to Kitsune's kit.cpu.num.threads intrinsic. On GPU's,
// this inserts a call to Kitsune's kit.gpu.num.compute.units.
Value *
PrepareReductionLoop::computeNumPartialReductions(Loop &outerLoop,
                                                  const Loop &innerLoop) {
  auto sanityCheck = [](const Loop &outerLoop, const Loop &innerLoop) {
    assert(outerLoop.getLoopPreheader() && "Outer loop must have a preheader");
    assert(hasTargetAttr(innerLoop) &&
           "Inner loop must have tapir target attribute");
  };

  LLVM_DEBUG(
      dbgs() << "PrepareReduction: Compute number of partial reductions\n");
  sanityCheck(outerLoop, innerLoop);

  BasicBlock &bb = *outerLoop.getLoopPreheader();
  LLVMContext &ctx = bb.getContext();

  TTID tt = *getTargetAttr(innerLoop);
  Intrinsic::ID numPartialsFunc = isGPUTT(tt)
                                      ? Intrinsic::kit_gpu_num_compute_units
                                      : Intrinsic::kit_cpu_num_threads;
  Constant *ctt = toConstant(tt, ctx);

  IRBuilder<> builder(bb.getTerminator());
  Value *numPartials =
      builder.CreateIntrinsic(numPartialsFunc, {ctt},
                              /*FMFSource=*/{}, "prduc.num.partials");

  return numPartials;
}

// Allocate a buffer into which a partial reduction will be accumulated. The
// buffer will be allocated in the inner loop preheader. This will also
// initialize the allocated buffer with the unit value for the reduction.
Value *PrepareReductionLoop::allocPartial(Loop &innerLoop,
                                          const ReductionInfo &info) {
  BasicBlock &bb = *innerLoop.getLoopPreheader();
  LLVMContext &ctx = bb.getContext();
  Type *i64 = Type::getInt64Ty(ctx);

  Value *unit = info.unit;
  Type *elemTy = unit->getType();
  Value *elemSize = info.elemSize;

  // FIXME: Instead of generating a call to malloc, we should generate a call
  // to a memory allocation intrinsic.
  IRBuilder builder(bb.getTerminator());
  Value *sz64 = builder.CreateIntCast(elemSize, i64, /*isSigned=*/true);
  Value *partial = builder.CreateMalloc(i64, elemTy, sz64, nullptr);
  builder.CreateStore(unit, partial);

  return partial;
}

// Change the destination of the reduction to accumulate into the buffer
// allocated for the partial reduction. Essentially, this changes the
// destination argument of the reduction intrinsic call.
void PrepareReductionLoop::reduceIntoPartial(Loop &innerLoop, Value *partial,
                                             const ReductionInfo &info) {
  CallBase *call = info.call;

  IRBuilder<> builder(call);
  Value *cst =
      builder.CreateAddrSpaceCast(partial, call->getArgOperand(2)->getType());
  call->setArgOperand(2, cst);
}

// Once the partial reduction has been computed, reduce this into the final
// destination of the reduction. This is done using LLVM's AtomicRMW instruction
// when it supports the reduction operator, or a custom atomic operation if it
// does not.
void PrepareReductionLoop::reducePartialIntoFinal(Loop &innerLoop,
                                                  Value *partial,
                                                  const ReductionInfo &info) {
  Value *dest = info.dest;
  Type *elemType = info.unit->getType();
  std::optional<AtomicRMWInst::BinOp> atomicOp = getAtomicOp(info.reduceOp);
  if (!atomicOp)
    emitDiagnostic(innerLoop, DiagID::ErrNYI,
                   "Reduction operators not supported by AtomicRMWInst");

  Module &m = *getModule(innerLoop);
  const DataLayout &dl = m.getDataLayout();
  Align align = dl.getPointerABIAlignment(KitAS::Default);

  BasicBlock &bb = *getExitBlockFromLatch(innerLoop);
  IRBuilder<> builder(bb.getTerminator());

  Value *v = builder.CreateLoad(elemType, partial);
  builder.CreateAtomicRMW(*atomicOp, dest, v, align, AtomicOrdering::Monotonic);
}

// Free the buffer allocated for the partial reduction.
void PrepareReductionLoop::freePartial(Loop &innerLoop, Value *partial) {
  BasicBlock &bb = *getExitBlockFromLatch(innerLoop);
  IRBuilder<> builder(bb.getTerminator());

  // FIXME: Replace this with a call to a memory deallocation intrinsic.
  builder.CreateFree(partial);
}

// When the outer loop was generated, the induction variable was canonical and
// in the range [0, n] in steps of 1 where `n` was the trip count of the tapir
// reduction loop being transformed. This must be changed so the range of the
// IV is [0, numPartials] where numPartials is the number of partial reductions
// to perform. The step will remain 1.
void PrepareReductionLoop::updateOuterLoopIV(Loop &outerLoop,
                                             Value *numPartials) {
  auto sanityCheck = [](const Loop &outerLoop, ScalarEvolution &se) {
    assert(outerLoop.isLoopSimplifyForm() &&
           "Outer loop must be in loop-simplify form");
    assert(outerLoop.getCanonicalInductionVariable() &&
           "Outer loop must have a canonical induction variable");
    assert(outerLoop.getBounds(se).has_value() &&
           "Outer loop must have computable loop bounds");
  };

  LLVM_DEBUG(
      dbgs() << "PrepareReduction: Update outer loop induction variable\n");
  sanityCheck(outerLoop, se);

  // The arguments to the comparison instruction will be the step instruction,
  // which increments the induction variable, and the final value. We just
  // change the operand that is not the step to be the new final value.
  PHINode *iv = outerLoop.getCanonicalInductionVariable();
  ICmpInst *cmp = outerLoop.getLatchCmpInst();
  Instruction *inc = &outerLoop.getBounds(se)->getStepInst();

  replaceNonMatchingOperands(*cmp, inc, numPartials);

  // Update analyses that may have been invalidated. We can't recalculate SE,
  // but we can force it recompute the analyses for certain variables and loops
  // when they are next requested.
  se.forgetValue(iv);
  se.forgetValue(cmp);
  se.forgetLoop(&outerLoop);

#ifndef NDEBUG
  se.verify();
#endif // NDEBUG
}

// For CPU's, a reduction loop of the form:
//
//     parallel_for (int i = 0; i < n; ++i)
//         a += ...
//
// is transformed into
//
//     parallel_for (int j = 0; j < numPartials; ++j) {
//         partialSize = (n + numPartials - 1) / numPartials
//         start = j * partialSize;
//         end = min(start + partialSize, n)
//         for (int i = start; i < end; ++i)
//             partials[j] += ...
//     }
//
// This function only changes the lower bound, upper bound and step of the
// inner loop.
void PrepareReductionLoop::updateInnerLoopIVCPU(Loop &outerLoop,
                                                Loop &innerLoop,
                                                Value *numPartials) {
  auto sanityCheck = [](const Loop &outerLoop, const Loop &innerLoop,
                        ScalarEvolution &se) {
    assert(outerLoop.getInductionVariable(se) &&
           "Outer loop must have an induction variable");
    assert(innerLoop.isLoopSimplifyForm() &&
           "Inner loop must be in loop-simplify form");
    assert(innerLoop.getCanonicalInductionVariable() &&
           "Inner loop must have a canonical induction variable");
    assert(innerLoop.getBounds(se).has_value() &&
           "Inner loop must have computable loop bounds");
    assert(innerLoop.getLoopPreheader() && "Inner loop must have a preheader");
    assert(getExitBlockFromLatch(innerLoop) &&
           "Inner loop must have a unique non-dead-end exit block");

    assert(pred_size(innerLoop.getLoopPreheader()) == 1 &&
           "Inner loop preheader must have a single predecessor");
    assert(succ_size(getExitBlockFromLatch(innerLoop)) == 1 &&
           "Inner loop exit block must have a single successor");
  };

  LLVM_DEBUG(
      dbgs()
      << "PrepareReduction: Update inner loop induction variable (CPU)\n");
  sanityCheck(outerLoop, innerLoop, se);

  PHINode *outerIV = outerLoop.getInductionVariable(se);
  Type *ivTy = outerIV->getType();

  Loop::LoopBounds innerBounds = *innerLoop.getBounds(se);
  PHINode *iv = innerLoop.getCanonicalInductionVariable();
  Value *tc = &innerBounds.getFinalIVValue();
  ICmpInst *cmp = innerLoop.getLatchCmpInst();
  BasicBlock *ph = innerLoop.getLoopPreheader();
  BasicBlock *guard = *pred_begin(ph);

  IRBuilder<> builder(&*guard->begin());
  Constant *one = ConstantInt::get(ivTy, 1, /*isSigned=*/false);
  Value *tcPlusPartials = builder.CreateAdd(tc, numPartials);
  Value *tcPlusPartialsSub1 = builder.CreateSub(tcPlusPartials, one);
  Value *sizePartials = builder.CreateUDiv(tcPlusPartialsSub1, numPartials,
                                           "prduc.size.partials");
  Value *newStart = builder.CreateMul(outerIV, sizePartials, "prduc.start");
  Value *newMax = builder.CreateAdd(newStart, sizePartials);
  Value *newEnd = builder.CreateIntrinsic(Intrinsic::umin, {ivTy}, {newMax, tc},
                                          /*FMFSource=*/{}, "prduc.end");

  iv->setIncomingValueForBlock(ph, newStart);
  replaceMatchingOperands(*cmp, tc, newEnd);

  se.forgetValue(iv);
  se.forgetValue(tc);
  se.forgetLoop(&innerLoop);

#ifndef NDEBUG
  dt.verify();
  se.verify();
#endif // NDEBUG
}

// For GPU's, a reduction loop of the form:
//
//     parallel_for (int i = 0; i < n; ++i)
//         a += ...
//
// is transformed into
//
//     parallel_for (int j = 0; j < numPartials; ++j)
//         for (int i = j; i < n; i += numPartials)
//             partials[j] += ...
//
// This function only changes the lower bound, upper bound and step of the
// inner loop.
void PrepareReductionLoop::updateInnerLoopIVGPU(Loop &outerLoop,
                                                Loop &innerLoop,
                                                Value *numPartials) {
  auto sanityCheck = [](const Loop &outerLoop, const Loop &innerLoop,
                        ScalarEvolution &se) {
    assert(outerLoop.getInductionVariable(se) &&
           "Outer loop must have an induction variable");
    assert(innerLoop.isLoopSimplifyForm() &&
           "Inner loop must be in loop-simplify form");
    assert(innerLoop.getCanonicalInductionVariable() &&
           "Inner loop must have a canonical induction variable");
    assert(innerLoop.getBounds(se).has_value() &&
           "Inner loop must have computable loop bounds");

    CmpInst *cmp = innerLoop.getLatchCmpInst();
    assert(cmp && "Inner loop must have a unique latch compare instruction");
    assert(cmp->getPredicate() == ICmpInst::ICMP_EQ &&
           "Expected inner loop comparison to be EQ");

    BasicBlock *latch = innerLoop.getLoopLatch();
    BranchInst *br = dyn_cast<BranchInst>(latch->getTerminator());
    assert(br && "Inner loop latch terminator must be a branch instruction");
    assert(br->getSuccessor(1) == innerLoop.getHeader() &&
           "Second successor of inner loop latch must be loop header");
  };

  LLVM_DEBUG(
      dbgs()
      << "PrepareReduction: Update inner loop induction variable (GPU)\n");
  sanityCheck(outerLoop, innerLoop, se);

  PHINode *outerIV = outerLoop.getInductionVariable(se);

  BasicBlock *innerPh = innerLoop.getLoopPreheader();
  PHINode *innerIV = innerLoop.getCanonicalInductionVariable();
  Instruction *innerStep = &innerLoop.getBounds(se)->getStepInst();
  ICmpInst *innerCmp = innerLoop.getLatchCmpInst();

  innerIV->setIncomingValueForBlock(innerPh, outerIV);
  replaceNonMatchingOperands(*innerStep, innerIV, numPartials);

  // The predicate of the compare instruction typically checks if the value of
  // loop induction variable is equal to the trip count and exits if it is. This
  // works because the primary loop induction variable is canonical. However,
  // since the inner loop may now have a non-unit step, it may go past the trip
  // count.
  //
  // The sanity check has already ensured that the false branch is the backedge
  // and the predicate is EQ. If the LHS of the comparison operand is the inner
  // step, then the comparison is effectively:
  //
  //     if (i + 1 == N) goto EXIT else goto HEADER
  //
  // We can then replace it with
  //
  //     if (i + 1 >= N) goto EXIT else goto HEADER
  //
  // Otherwise, the comparison and the replacements are as shown below
  //
  //     if (N == i + 1) goto EXIT else goto HEADER
  //     if (N <= i + 1) goto EXIT else goto HEADER
  //
  if (innerCmp->getOperand(0) == innerStep)
    innerCmp->setPredicate(ICmpInst::ICMP_UGE);
  else
    innerCmp->setPredicate(ICmpInst::ICMP_ULE);

  se.forgetValue(innerIV);
  se.forgetValue(innerStep);
  se.forgetValue(innerCmp);
  se.forgetLoop(&innerLoop);

#ifndef NDEBUG
  dt.verify();
  li.verify(dt);
  se.verify();
#endif // NDEBUG
}

void PrepareReductionLoop::updateInnerLoopGuardCondition(Loop &loop,
                                                         Value *numPartials) {
  auto sanityCheck = [](Loop &loop, BranchInst *br, ScalarEvolution &se) {
    assert(br && "Inner loop must have a guard block");
    assert(isFalse(br->getCondition()) &&
           "Temporary condition of inner loop guard branch must be `false`");

    assert(loop.getBounds(se).has_value() &&
           "Inner loop must have finite bounds");
  };

  LLVM_DEBUG(dbgs() << "PrepareReduction: Update inner loop guard condition\n");
  sanityCheck(loop, getWrappedLoopGuardBranch(loop), se);

  Loop::LoopBounds bounds = *loop.getBounds(se);
  Value *start = &bounds.getInitialIVValue();
  Value *end = &bounds.getFinalIVValue();

  BranchInst *br = getWrappedLoopGuardBranch(loop);
  Value *cmp = new ICmpInst(br->getIterator(), ICmpInst::ICMP_UGE, start, end);

  br->setCondition(cmp);

  se.forgetValue(br);

#ifndef NDEBUG
  se.verify();
#endif // NDEBUG
}

void PrepareReductionLoop::parallelizeOuterLoop(Loop &outerLoop, Loop &loop) {
  auto sanityCheck = [](const Loop &outerLoop, const Loop &loop, TaskInfo &ti) {
    BasicBlock *innerPh = loop.getLoopPreheader();
    BasicBlock *innerExit = getExitBlockFromLatch(loop);
    BasicBlock *outerHeader = outerLoop.getHeader();
    BasicBlock *outerLatch = outerLoop.getLoopLatch();

    assert(innerPh && "Inner loop must have a preheader");
    assert(innerExit && "Inner loop must have a unique exit block");
    assert(outerLatch && "Outer loop must have a unique latch");

    assert(pred_size(innerPh) == 1 &&
           "Inner loop preheader must have a single predecessor");
    BasicBlock *innerGuard = *pred_begin(innerPh);

    assert(succ_size(outerHeader) == 1 &&
           "Outer loop header must have a single successor");
    assert(*succ_begin(outerHeader) == innerGuard &&
           "Successor of outer loop header must be inner loop guard");

    assert(succ_size(innerExit) == 1 &&
           "Inner loop exit block must have a single successor");
    BasicBlock *innerEnd = *succ_begin(innerExit);

    assert(succ_size(innerEnd) == 1 &&
           "Inner loop end block must have a single successor");
    assert(pred_size(outerLatch) == 1 &&
           "Outer loop latch must have a single predecessor");
    assert(*pred_begin(outerLatch) == *succ_begin(innerEnd) &&
           "Predecessor of outer loop latch must be the successor of the inner "
           "loop end block");

    // The outer loop should be parallelized before the inner loop, so the
    // inner loop should be still recognized as a tapir loop here.
    assert(getTaskIfTapirLoop(&loop, &ti) &&
           "Inner loop not recognized as a tapir loop");
  };

  LLVM_DEBUG(dbgs() << "PrepareReduction: Parallelize outer loop\n");
  sanityCheck(outerLoop, loop, ti);

  Value *syncRegion = getTapirLoopSyncRegion(loop);

  BasicBlock *latch = outerLoop.getLoopLatch();
  BasicBlock *reattach = *pred_begin(latch);
  ReattachInst *reattachInst = ReattachInst::Create(latch, syncRegion);
  ReplaceInstWithInst(reattach->getTerminator(), reattachInst);

  BasicBlock *header = outerLoop.getHeader();
  BasicBlock *innerPh = loop.getLoopPreheader();
  BasicBlock *innerGuard = *pred_begin(innerPh);
  DetachInst *detachInst = DetachInst::Create(innerGuard, latch, syncRegion);
  ReplaceInstWithInst(header->getTerminator(), detachInst);

  // Update analyses that may have been invalidated by this transformation.
  dt.insertEdge(header, latch);

#ifndef NDEBUG
  dt.verify();
#endif // NDEBUG

  // FIXME: Recalculating everything in TaskInfo is wasteful, but it is not
  // clear that there is a better way to do it currently.
  Function *f = getFunction(loop);
  ti.recalculate(*f, dt);

#ifndef NDEBUG
  ti.verify(dt);

  // Since the outer loop should be parallelized before the inner loop, both
  // should be recognized as parallel here.
  assert(getTaskIfTapirLoop(&outerLoop, &ti) &&
         "Outer loop recognized as tapir loop after parallelization");
  assert(
      getTaskIfTapirLoop(&loop, &ti) &&
      "Inner loop recognized as tapir loop after outer loop was parallelized");
#endif // NDEBUG
}

void PrepareReductionLoop::serializeInnerLoop(Loop &loop) {
  auto sanityCheck = [](Loop &loop, TaskInfo &ti) {
    assert(getTaskIfTapirLoop(&loop, &ti) && "Getting task from tapir loop");
  };

  LLVM_DEBUG(dbgs() << "PrepareReduction: Serialize inner loop\n");
  sanityCheck(loop, ti);

  Task *task = getTaskIfTapirLoop(&loop, &ti);
  serializeTapirLoop(loop, *task, /*addSerializedAttr=*/false, &dt, &ti);

  // serializeTapirLoop will already have updated the analyses that were
  // invalidated by this transformation

  // FIXME: Recalculating everything in TaskInfo is wasteful, but it is not
  // clear that there is a better way to do it currently.
  Function *f = getFunction(loop);
  ti.recalculate(*f, dt);

#ifndef DEBUG
  ti.verify(dt);
  assert(!getTaskIfTapirLoop(&loop, &ti) &&
         "Inner loop not recognized as tapir loop after serialization");
#endif // NDEBUG
}

// Transform a tapir reduction loop. Add the tapir.loop.prepared attribute to
// the loop and return true if anything other than this attribute was changed.
// For example, if the loop does not perform any actual reductions, the
// attribute will be added to the loop, but no other transformations will be
// performed. In such cases, simply return false.
bool PrepareReductionLoop::run(TapirLoopInfo &tapirLoop) {
  auto sanityCheck = [](const TapirLoopInfo &tapirLoop) {
    const Loop &loop = *tapirLoop.getLoop();

    // These checks are mostly to ensure that the loop objects known to the
    // driver don't get corrupted when processing a function with many tapir
    // loops.
    assert(isTapirLoop(loop) &&
           "Loop with reduction attribute is a tapir loop");
    assert(hasReductionAttr(loop) &&
           "Loop is expected to have the reduction attribute");
    assert(!hasPreparedAttr(loop) &&
           "Tapir reduction loop has not been prepared");
  };

  LLVM_DEBUG(dbgs() << "PrepareReduction: BEGIN '"
                    << getName(*tapirLoop.getLoop()) << "'\n");
  sanityCheck(tapirLoop);

  Loop &loop = *tapirLoop.getLoop();

#ifndef NDEBUG
  // Make sure that the parent loop, if any, is not affected by this
  // transformation.
  Loop *parentLoop = loop.getParentLoop();
#endif // NDEBUG

  // If the loop does not perform any actual reductions, mark it as prepared,
  // but return false because only the metadata will have changed.
  SmallVector<ReductionInfo, 1> reductions = collectReductions(loop);
  if (reductions.empty()) {
    addPreparedAttr(loop);
    return false;
  }

  // Generate the outer loop including all blocks. The analysis objects will
  // have been updated.
  Loop *outerLoop = wrapWithTapirLoop(tapirLoop, dt, li, mssa);

  // Don't make any changes to the loop trip counts just yet. While the
  // transformations to the reduction loop should not be affected by any change
  // to the trip count, it may be safer to leave them as canonical until we have
  // finished all the other transformations.

  // This inserts code for the allocation, initialization, use and deallocation
  // of the partial reduction buffers. Most of these only involve inserting
  // calls to Kitsune-specific intrinsic and other relatively simple
  // instructions to dedicated basic blocks.
  Value *numPartials = computeNumPartialReductions(*outerLoop, loop);
  for (const ReductionInfo &reduction : reductions) {
    Value *partial = allocPartial(loop, reduction);
    reduceIntoPartial(loop, partial, reduction);
    reducePartialIntoFinal(loop, partial, reduction);
    freePartial(loop, partial);
  }

  // Now that everything else has been changed, we can fix up the induction
  // variables of the outer and inner loops. These will only change the initial
  // value, final value and step of the IV, but should not change the IV itself.
  // We do this late to minimize the analyses that need to be recomputed
  updateOuterLoopIV(*outerLoop, numPartials);

  TTID tt = *getTargetAttr(loop);
  if (isGPUTT(tt))
    updateInnerLoopIVGPU(*outerLoop, loop, numPartials);
  else
    updateInnerLoopIVCPU(*outerLoop, loop, numPartials);
  updateInnerLoopGuardCondition(loop, numPartials);

  // The outer loop is serial and the inner is parallel. But it needs to be the
  // other way around. Once serializeInnerLoop is run, the syncregion associated
  // with it will be lost. This is required to parallelize the outer loop.
  // Therefore, the outer loop must be parallelized before the inner loop is
  // serialized.
  //
  // WARNING: Once we have run parallelizeOuterLoop and serializeInnerLoop, the
  // references to the tapir task objects will not be safe to use. This is why
  // this is done as late as possible - when we should no longer need any
  // objects from TaskInfo.
  parallelizeOuterLoop(*outerLoop, loop);
  serializeInnerLoop(loop);

#ifndef NDEBUG
  dt.verify();
  li.verify(dt);
  ti.verify(dt);

  loop.verifyLoop();
  outerLoop->verifyLoop();
  if (parentLoop) {
    parentLoop->verifyLoop();
    assert(outerLoop->getParentLoop() == parentLoop &&
           "Original parent of inner loop must now be parent of outer loop");
  }

  // We try to maintain the loops in simplify form. A later pass may well try to
  // simplify it, but we want to ensure that we preserve this. In case this pass
  // needs to be modified in the future, this may help ensure that it the
  // modifications are "consistent".
  assert(outerLoop->isLoopSimplifyForm() &&
         "Outer loop must be in loop-simplify form");
  assert(loop.isLoopSimplifyForm() &&
         "Inner loop must be in loop-simplify form");

  // The inner loop may not have had a guard. But one will have been added as
  // part of this transformation. Make sure that it was added correctly.
  BranchInst *br = getWrappedLoopGuardBranch(loop);
  assert(br && "Inner loop must have a guard");
  assert(br->getNumSuccessors() == 2 &&
         "Inner loop guard must have two successors");
  assert(br->getSuccessor(0) == *succ_begin(getExitBlockFromLatch(loop)) &&
         "First successor of inner loop guard must be inner loop end");
  assert(br->getSuccessor(1) == loop.getLoopPreheader() &&
         "Second successor of inner loop guard must be inner loop preheader");

  // The outer loop will have been parallelized. Make sure that this is done
  // correctly.
  assert(getTaskIfTapirLoop(outerLoop, &ti) &&
         "Outer loop recognized as tapir loop");

  // The inner loop will have been serialized. Make sure this is reflected in
  // the task analysis object.
  assert(!getTaskIfTapirLoop(&loop, &ti) &&
         "Inner loop not recognized as tapir loop");
#endif // NDEBUG

  LLVM_DEBUG(dbgs() << "PrepareReduction: END '" << getName(loop) << "'\n");

  // Mark the loop as having been prepared to ensure that we don't accidentally
  // attempt to process it more than once.
  addPreparedAttr(*outerLoop);
  return true;
}

template <typename... Args>
static bool complain(const Loop &loop, DiagID diag, Args &&...args) {
  emitDiagnostic(loop, diag, args...);
  return false;
}

static bool prepareForCPU(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                          LoopInfo &li, MemorySSA &mssa, ScalarEvolution &se,
                          TaskInfo &ti) {
  return PrepareReductionLoop(dt, li, mssa, se, ti).run(tapirLoop);
}

static bool prepareForGPU(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                          LoopInfo &li, MemorySSA &mssa, ScalarEvolution &se,
                          TaskInfo &ti) {
  return PrepareReductionLoop(dt, li, mssa, se, ti).run(tapirLoop);
}

static bool prepareForSerial(TapirLoopInfo &tapirLoop) {
  // There is nothing to be done to prepare a tapir reduction loop for the
  // serial tapir target. Calls to Kitsune's reduce intrinsics will be lowered
  // in a separate pass.
  addPreparedAttr(*tapirLoop.getLoop());
  return false;
}

bool llvm::prepareReductionLoop(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                                LoopInfo &li, MemorySSA &mssa,
                                ScalarEvolution &se, TaskInfo &ti) {
  Loop &loop = *tapirLoop.getLoop();
  TTID tt = *getTargetAttr(loop);
  if (tt == TTID::Serial)
    return prepareForSerial(tapirLoop);
  else if (isCPUTT(tt))
    return prepareForCPU(tapirLoop, dt, li, mssa, se, ti);
  else if (isGPUTT(tt))
    return prepareForGPU(tapirLoop, dt, li, mssa, se, ti);
  else
    return false;
}

bool llvm::checkReductionLoop(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                              LoopInfo &li) {
  if (!checkTapirLoopSafeToWrap(tapirLoop, dt, li))
    return false;

  // If this is not a top-level tapir loop, it is nested within another tapir
  // loop.
  // FIXME: Add support for reductions in nested tapir loops.
  const Loop &loop = *tapirLoop.getLoop();
  if (!isTopLevelTapirLoop(loop))
    return complain(loop, DiagID::ErrGeneric,
                    "tapir reduction loop must be a top-level loop");

  return true;
}

//===- PrepareParallelLoops.cpp - Transform non-reduction tapir loops -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transform tapir loops that do not perform reductions to a form that is
// suitable for parallel execution.
//
// Consider the loop shown below
//
//     parallel_for (int i = 0; i < n; ++i) {
//         ...
//     }
//
// This pass will transform this into the following for parallel execution on a
// CPU.
//
//     int64_t numThreads = kit.cpu.num.threads();
//     int64_t itersPerThread = (n + numThreads - 1) / numThreads;
//     parallel_for (int j = 0; j < numThreads; ++j) {
//         int start = j * itersPerThread;
//         int end = std::min(start + itersPerThread, n);
//         for (int i = start; i < end; ++i) {
//             ...
//         }
//     }
//
// For GPU's, on the other hand, the loop will remain unchanged.
//
// The main reason for this transformation is to enable vectorization in
// parallel loops. Currently, LLVM's vectorizer is not able to reason about
// tapir loops since it cannot determine the correctness of vectorization in
// the presence of tapir instructions. With this, and subsequent transformations
// that will convert the inner loop to have a canonical induction variable, the
// inner loop can be vectorized.
//
// The other advantage of this transformation is that it simplifies the
// implementation of Kitsune's CPU-centric runtimes since they no longer need to
// determine how many iterations of the parallel loop are to be performed on
// each thread - every thread will perform exactly one iteration.
//
// ===---------------------------------------------------------------------===//

#include "PrepareParallelLoops.h"
#include "LoopWrapping.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/InstUtils.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
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

/// Base class to transform tapir loops for parallel execution. The default
/// implementation is suitable for Kitsune's CPU-centric parallel tapir targets,
/// such as openmp, pthreads, and qthreads. It may also be used with opencilk.
/// Other tapir targets may need to specialize this.
class PrepareParallelLoopBase {
private:
  DominatorTree &dt;
  LoopInfo &li;
  MemorySSA &mssa;
  ScalarEvolution &se;
  TaskInfo &ti;

  DomTreeUpdater dtu;
  MemorySSAUpdater mssau;

private:
  void updateOuterLoopIV(Loop &outerLoop, Value *numThreads);
  void updateInnerLoopIV(Loop &outerLoop, Loop &innerLoop, Value *numThreads);
  void updateInnerLoopGuardCondition(Loop &innerLoop);
  void parallelizeOuterLoop(Loop &outerLoop, Loop &loop);
  void serializeInnerLoop(Loop &loop);

protected:
  virtual Value *computeNumCPUThreads(BasicBlock &bb,
                                      const TapirLoopInfo &loop);

  PrepareParallelLoopBase(DominatorTree &dt, LoopInfo &li, MemorySSA &mssa,
                          ScalarEvolution &se, TaskInfo &ti)
      : dt(dt), li(li), mssa(mssa), se(se), ti(ti),
        dtu(dt, DomTreeUpdater::UpdateStrategy::Eager), mssau(&mssa) {}

public:
  bool run(TapirLoopInfo &tapirLoop);
  virtual ~PrepareParallelLoopBase() = default;
};

/// Class to transform non-reduction tapir loops for parallel execution for
/// Kitsune's CPU-centric parallel runtimes.
class PrepareParallelLoopCPU : public PrepareParallelLoopBase {
public:
  PrepareParallelLoopCPU(DominatorTree &dt, LoopInfo &li, MemorySSA &mssa,
                         ScalarEvolution &se, TaskInfo &ti)
      : PrepareParallelLoopBase(dt, li, mssa, se, ti) {}
  virtual ~PrepareParallelLoopCPU() = default;
};

/// Class to transform non-reduction tapir loops with the serial tapir target.
class PrepareParallelLoopSerial : public PrepareParallelLoopBase {
protected:
  virtual Value *computeNumCPUThreads(BasicBlock &bb,
                                      const TapirLoopInfo &loop) override;

public:
  PrepareParallelLoopSerial(DominatorTree &dt, LoopInfo &li, MemorySSA &mssa,
                            ScalarEvolution &se, TaskInfo &ti)
      : PrepareParallelLoopBase(dt, li, mssa, se, ti) {}
  virtual ~PrepareParallelLoopSerial() = default;
};

} // namespace

// Insert code to calculate the number of CPU threads available for use. This
// simply inserts a call to Kitsune's kit.cpu.num.threads intrinsic. The
// intrinsic will be lowered in a later pass.
//
//      numThreads = call @llvm.kit.cpu.num.threads()
//
Value *
PrepareParallelLoopBase::computeNumCPUThreads(BasicBlock &bb,
                                              const TapirLoopInfo &tapirLoop) {
  auto sanityCheck = [](const TapirLoopInfo &tapirLoop) {
    assert(hasTargetAttr(*tapirLoop.getLoop()) &&
           "Outer loop must have tapir target attribute");
  };

  LLVM_DEBUG(dbgs() << "PrepareParallel: Get the number of parallel workers\n");
  sanityCheck(tapirLoop);

  LLVMContext &ctx = bb.getContext();

  Loop &loop = *tapirLoop.getLoop();
  TTID tt = *getTargetAttr(loop);
  Constant *ctt = toConstant(tt, ctx);

  PHINode *iv = loop.getCanonicalInductionVariable();
  Type *ivTy = iv->getType();

  IRBuilder<> builder(&*bb.begin());
  Value *thrds = builder.CreateIntrinsic(Intrinsic::kit_cpu_num_threads, {ctt});
  Value *numThreads =
      builder.CreateIntCast(thrds, ivTy, /*isSigned=*/true, "prll.num.threads");

  return numThreads;
}

// When the outer loop was generated, the induction variable was canonical and
// in the range [0, n] in steps of 1 where `n` was the trip count of the tapir
// loop being transformed. This must be changed so the range of the IV is [0,
// numThreads] where numThreads is the number of CPU threads available. The step
// will remain 1.
void PrepareParallelLoopBase::updateOuterLoopIV(Loop &outerLoop,
                                                Value *numThreads) {
  auto sanityCheck = [](const Loop &outerLoop, ScalarEvolution &se) {
    assert(outerLoop.isLoopSimplifyForm() &&
           "Outer loop must be in loop-simplify form");
    assert(outerLoop.getCanonicalInductionVariable() &&
           "Outer loop must have a canonical induction variable");
    assert(outerLoop.getBounds(se).has_value() &&
           "Outer loop must have computable loop bounds");
  };

  LLVM_DEBUG(
      dbgs() << "PrepareParallel: Update outer loop induction variable\n");
  sanityCheck(outerLoop, se);

  // The arguments to the comparison instruction will be the step instruction,
  // which increments the induction variable, and the final value. We just
  // change the operand that is not the step to be the new final value.
  PHINode *iv = outerLoop.getCanonicalInductionVariable();
  ICmpInst *cmp = outerLoop.getLatchCmpInst();
  Instruction *inc = &outerLoop.getBounds(se)->getStepInst();

  replaceNonMatchingOperands(*cmp, inc, numThreads);

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

// For CPU's, a loop of the form:
//
//     parallel_for (int i = 0; i < n; ++i)
//         ...
//
// is transformed into
//
//     parallel_for (int j = 0; j < numThreads; ++j) {
//         itersPerThread = (n + numThreads - 1) / numThreads
//         start = j * itersPerThread;
//         end = min(start + itersPerThread, n)
//         for (int i = start; i < end; ++i)
//             ...
//     }
//
// This function only changes the lower bound, upper bound and step of the
// inner loop.
void PrepareParallelLoopBase::updateInnerLoopIV(Loop &outerLoop,
                                                Loop &innerLoop,
                                                Value *numThreads) {
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
      dbgs() << "PrepareParallel: Update inner loop induction variable\n");
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
  Value *tcPlusThreads = builder.CreateAdd(tc, numThreads);
  Value *tcPlusThreadsSub1 = builder.CreateSub(tcPlusThreads, one);
  Value *itersPerThread =
      builder.CreateUDiv(tcPlusThreadsSub1, numThreads, "prll.per.thrd");
  Value *newStart = builder.CreateMul(outerIV, itersPerThread, "prll.start");
  Value *newMax = builder.CreateAdd(newStart, itersPerThread);
  Value *newEnd = builder.CreateIntrinsic(Intrinsic::umin, {ivTy}, {newMax, tc},
                                          /*FMFSource=*/{}, "prll.end");

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

void PrepareParallelLoopBase::updateInnerLoopGuardCondition(Loop &loop) {
  auto sanityCheck = [](Loop &loop, BranchInst *br, ScalarEvolution &se) {
    assert(br && "Inner loop must have a guard block");
    assert(isFalse(br->getCondition()) &&
           "Placeholder condition of inner loop guard branch must be `false`");

    assert(loop.getBounds(se).has_value() &&
           "Inner loop must have finite bounds");
  };

  LLVM_DEBUG(dbgs() << "PrepareParallel: Update inner loop guard condition\n");
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

void PrepareParallelLoopBase::parallelizeOuterLoop(Loop &outerLoop,
                                                   Loop &loop) {
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

  LLVM_DEBUG(dbgs() << "PrepareParallel: Parallelize outer loop\n");
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

void PrepareParallelLoopBase::serializeInnerLoop(Loop &loop) {
  auto sanityCheck = [](Loop &loop, TaskInfo &ti) {
    assert(getTaskIfTapirLoop(&loop, &ti) && "Getting task from tapir loop");
  };

  LLVM_DEBUG(dbgs() << "PrepareParallel: Serialize inner loop\n");
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

// Transform a tapir loop. Add the tapir.loop.prepared attribute to the loop and
// return true if anything other than this attribute was changed.
bool PrepareParallelLoopBase::run(TapirLoopInfo &tapirLoop) {
  auto sanityCheck = [](const TapirLoopInfo &tapirLoop) {
    const Loop &loop = *tapirLoop.getLoop();

    // These checks are mostly to ensure that the loop objects known to the
    // driver don't get corrupted when processing a function with many tapir
    // loops.
    assert(isTapirLoop(loop) && "Loop is a tapir loop");
    assert(!hasPreparedAttr(loop) && "Tapir loop has not been prepared");
  };

  LLVM_DEBUG(dbgs() << "PrepareParallel: BEGIN '"
                    << getName(*tapirLoop.getLoop()) << "'\n");
  sanityCheck(tapirLoop);

  Loop &loop = *tapirLoop.getLoop();

#ifndef NDEBUG
  // Make sure that the parent loop, if any, is not affected by this
  // transformation.
  Loop *parentLoop = loop.getParentLoop();
#endif // NDEBUG

  // Generate the outer loop including all blocks. The analysis objects will
  // have been updated.
  Loop *outerLoop = wrapWithTapirLoop(tapirLoop, dt, li, mssa);
  BasicBlock *outerLoopPreheader = outerLoop->getLoopPreheader();
  Value *numThreads = computeNumCPUThreads(*outerLoopPreheader, tapirLoop);

  // Fix up the induction variables of the outer and inner loops. These will
  // only change the initial value, final value and step of the IV, but should
  // not change the IV itself.
  updateOuterLoopIV(*outerLoop, numThreads);
  updateInnerLoopIV(*outerLoop, loop, numThreads);
  updateInnerLoopGuardCondition(loop);

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

  LLVM_DEBUG(dbgs() << "PrepareParallel: END '" << getName(loop) << "'\n");

  // Mark the loop as having been prepared to ensure that we don't accidentally
  // attempt to process it more than once.
  addPreparedAttr(*outerLoop);
  return true;
}

Value *PrepareParallelLoopSerial::computeNumCPUThreads(BasicBlock &bb,
                                                       const TapirLoopInfo &) {
  return toConstant(1L, bb.getContext());
}

static bool prepareForCPU(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                          LoopInfo &li, MemorySSA &mssa, ScalarEvolution &se,
                          TaskInfo &ti) {
  return PrepareParallelLoopCPU(dt, li, mssa, se, ti).run(tapirLoop);
}

static bool prepareForGPU(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                          LoopInfo &li, MemorySSA &mssa, ScalarEvolution &se,
                          TaskInfo &ti) {
  // We currently do not perform any transformation for parallel loops to be
  // executed on the GPU. The main purpose of the CPU-side transformation is to
  // enable vectorization. It is not clear if this is beneficial, or even
  // possible, on GPU's. In the future, if there is some transformation is
  // beneficial, that can be carried out here.
  addPreparedAttr(*tapirLoop.getLoop());
  return false;
}

static bool prepareForSerial(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                             LoopInfo &li, MemorySSA &mssa, ScalarEvolution &se,
                             TaskInfo &ti) {
  // We could just serialize the loop since that will be equivalent. Still, we
  // perform a comparable transformation because passes later in the pipeline
  // may want to do something with loops that have been transformed this way.
  // One problem with this approach though, is that the outer loop will almost
  // certainly be DCE'ed since the optimizer can clearly see that the trip count
  // is 1. This means that we will lose the "provenance" of this loop,
  // specifically that it was intended to be lowered using the serial tapir
  // target. Some passes, such as `kit-ctors`, need this provenance, The hack
  // for now, is to add the `tapir.loop.serialized` attribute to the inner
  // loop since that can be examined by the passes that need this information.
  bool prepared =
      PrepareParallelLoopSerial(dt, li, mssa, se, ti).run(tapirLoop);
  addSerializedAttr(*tapirLoop.getLoop());

  return prepared;
}

bool llvm::prepareParallelLoop(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                               LoopInfo &li, MemorySSA &mssa,
                               ScalarEvolution &se, TaskInfo &ti) {
  Loop &loop = *tapirLoop.getLoop();
  TTID tt = *getTargetAttr(loop);
  if (tt == TTID::Serial)
    return prepareForSerial(tapirLoop, dt, li, mssa, se, ti);
  else if (isCPUTT(tt))
    return prepareForCPU(tapirLoop, dt, li, mssa, se, ti);
  else if (isGPUTT(tt))
    return prepareForGPU(tapirLoop, dt, li, mssa, se, ti);
  else
    return false;
}

bool llvm::checkParallelLoop(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                             LoopInfo &li) {
  return checkTapirLoopSafeToWrap(tapirLoop, dt, li);
}

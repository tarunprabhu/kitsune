//===- PrepareTapirLoops.cpp - Prepare tapir loops for lowering -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transform tapir loops to a form suitable for parallel execution.
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
// For GPU's, the loop will be transformed into the following:
//
//     // The number of iterations of the parallel loop to be run on each thread
//     int64_t itersPerThread = kit.gpu.ipt();
//     int64_t numThreads = (n + itersPerThread - 1) / itersPerThread;
//     parallel_for (int j = 0; j < numThreads; ++j) {
//         int start = j * itersPerThread;
//         int end = std::min(start + itersPerThread, n);
//         for (int i = start; i < end; ++i) {
//             ...
//         }
//     }
//
// Note that, unlike the transformation for the CPU where we have a fixed number
// of parallel workers, for the GPU, we decide how much work to do per thread,
// then determine the number of threads to use from that. The kit.gpu.ipt
// intrinsic may be lowered to a compile-time constant, or a call to a runtime
// function.
//
// The other major difference in GPU-lowering is in the case of nested parallel
// loops. In the CPU case, each loop in the nest is transformed in exactly the
// same way. For GPU's, though, only the innermost loop is transformed. Consider
// the nested parallel loops below.
//
//     parallel_for (int i = 0; i < m; ++i) {
//         parallel_for (int j = 0; j < n; ++j) {
//             ...
//         }
//     }
//
// This will be transformed as shown below:
//
//     parallel_for (int i = 0; i < m; ++i) {
//       int64_t itersPerThread = kit.gpu.ipt();
//       int64_t numThreads = (n + itersPerThread - 1) / itersPerThread;
//       parallel_for (int j = 0; j < numThreads; ++j) {
//           int start = j * itersPerThread;
//           int end = std::min(start + itersPerThread, n);
//           for (int i = start; i < end; ++i) {
//               ...
//           }
//       }
//    }
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/PrepareTapirLoops.h"
#include "LoopWrapping.h"
#include "PrepareParallelLoops.h"
#include "PrepareReductionLoops.h"
#include "PrepareTapirLoopsImpl.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/InstUtils.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/ValueUtils.h"
#include "kitsune/Passes/PassUtils.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

#define DEBUG_TYPE "kit-prepare"

using namespace llvm;

// When the outer loop was generated, the induction variable was canonical and
// in the range [0, n] in steps of 1 where `n` was the trip count of the tapir
// loop being transformed. This must be changed so the range of the IV is [0,
// numThreads] where numThreads is the number of CPU threads available. The step
// will remain 1.
void detail::PrepareTapirLoopBase::updateOuterLoopIV(Loop &outerLoop,
                                                     Value *outerTC) {
  auto sanityCheck = [](const Loop &outerLoop, ScalarEvolution &se) {
    assert(outerLoop.isLoopSimplifyForm() &&
           "Outer loop must be in loop-simplify form");
    assert(outerLoop.getCanonicalInductionVariable() &&
           "Outer loop must have a canonical induction variable");
    assert(outerLoop.getBounds(se).has_value() &&
           "Outer loop must have computable loop bounds");
  };

  LLVM_DEBUG(dbgs() << "PrepareTapir: Update outer loop induction variable\n");
  sanityCheck(outerLoop, se);

  // The arguments to the comparison instruction will be the step instruction,
  // which increments the induction variable, and the final value. We just
  // change the operand that is not the step to be the new final value.
  PHINode *iv = outerLoop.getCanonicalInductionVariable();
  ICmpInst *cmp = outerLoop.getLatchCmpInst();
  Instruction *inc = &outerLoop.getBounds(se)->getStepInst();

  replaceNonMatchingOperands(*cmp, inc, outerTC);

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
// This function only changes the lower bound, upper bound and step of the inner
// loop.
void detail::PrepareTapirLoopBase::updateInnerLoopIV(Loop &outerLoop,
                                                     Loop &innerLoop,
                                                     Value *outerTC) {
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

  LLVM_DEBUG(dbgs() << "PrepareTapir: Update inner loop induction variable\n");
  sanityCheck(outerLoop, innerLoop, se);

  PHINode *outerIV = outerLoop.getInductionVariable(se);
  Type *ivTy = outerIV->getType();

  Loop::LoopBounds innerBounds = *innerLoop.getBounds(se);
  PHINode *iv = innerLoop.getCanonicalInductionVariable();
  Value *innerTC = &innerBounds.getFinalIVValue();
  ICmpInst *cmp = innerLoop.getLatchCmpInst();
  BasicBlock *ph = innerLoop.getLoopPreheader();
  BasicBlock *guard = *pred_begin(ph);

  IRBuilder<> builder(&*guard->begin());
  Constant *one = ConstantInt::get(ivTy, 1, /*isSigned=*/false);
  Value *iPlusO = builder.CreateAdd(innerTC, outerTC);
  Value *iPlusOMinus1 = builder.CreateSub(iPlusO, one);
  Value *itersPerThread =
      builder.CreateUDiv(iPlusOMinus1, outerTC, "prll.per.thrd");
  Value *newStart = builder.CreateMul(outerIV, itersPerThread, "prll.start");
  Value *newMax = builder.CreateAdd(newStart, itersPerThread);
  Value *newEnd =
      builder.CreateIntrinsic(Intrinsic::umin, {ivTy}, {newMax, innerTC},
                              /*FMFSource=*/{}, "prll.end");

  iv->setIncomingValueForBlock(ph, newStart);
  replaceMatchingOperands(*cmp, innerTC, newEnd);

  se.forgetValue(iv);
  se.forgetValue(innerTC);
  se.forgetLoop(&innerLoop);

#ifndef NDEBUG
  dt.verify();
  se.verify();
#endif // NDEBUG
}

void detail::PrepareTapirLoopBase::updateInnerLoopGuardCondition(Loop &loop) {
  auto sanityCheck = [](Loop &loop, BranchInst *br, ScalarEvolution &se) {
    assert(br && "Inner loop must have a guard block");
    assert(isFalse(br->getCondition()) &&
           "Placeholder condition of inner loop guard branch must be `false`");

    assert(loop.getBounds(se).has_value() &&
           "Inner loop must have finite bounds");
  };

  LLVM_DEBUG(dbgs() << "PrepareTapir: Update inner loop guard condition\n");
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

void detail::PrepareTapirLoopBase::parallelizeOuterLoop(Loop &outerLoop,
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

  LLVM_DEBUG(dbgs() << "PrepareTapir: Parallelize outer loop\n");
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

void detail::PrepareTapirLoopBase::serializeInnerLoop(Loop &loop) {
  auto sanityCheck = [](Loop &loop, TaskInfo &ti) {
    assert(getTaskIfTapirLoop(&loop, &ti) && "Getting task from tapir loop");
  };

  LLVM_DEBUG(dbgs() << "PrepareTapir: Serialize inner loop\n");
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
bool detail::PrepareTapirLoopBase::run(TapirLoopInfo &tapirLoop) {
  auto sanityCheck = [](const TapirLoopInfo &tapirLoop) {
    const Loop &loop = *tapirLoop.getLoop();

    // These checks are mostly to ensure that the loop objects known to the
    // driver don't get corrupted when processing a function with many tapir
    // loops.
    assert(isTapirLoop(loop) && "Loop is a tapir loop");
    assert(!hasPreparedAttr(loop) && "Tapir loop has not been prepared");
  };

  LLVM_DEBUG(dbgs() << "PrepareTapir: BEGIN '" << getName(*tapirLoop.getLoop())
                    << "'\n");
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
  Value *outerTC = computeOuterLoopTripCount(*outerLoopPreheader, tapirLoop);

  // Fix up the induction variables of the outer and inner loops. These will
  // only change the initial value, final value and step of the IV, but should
  // not change the IV itself.
  updateOuterLoopIV(*outerLoop, outerTC);
  updateInnerLoopIV(*outerLoop, loop, outerTC);
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

  LLVM_DEBUG(dbgs() << "PrepareTapir: END '" << getName(loop) << "'\n");

  // Mark the loop as having been prepared to ensure that we don't accidentally
  // attempt to process it more than once.
  addPreparedAttr(*outerLoop);
  return true;
}

//===----------------------------------------------------------------------===//

// Insert code to calculate the number of CPU threads available for use. This
// simply inserts a call to Kitsune's kit.cpu.num.threads intrinsic. The
// intrinsic will be lowered in a later pass.
//
//      numThreads = call @llvm.kit.cpu.num.threads()
//
Value *detail::PrepareTapirLoopCPU::computeOuterLoopTripCount(
    BasicBlock &bb, const TapirLoopInfo &tapirLoop) {
  auto sanityCheck = [](const TapirLoopInfo &tapirLoop) {
    assert(hasTargetAttr(*tapirLoop.getLoop()) &&
           "Outer loop must have tapir target attribute");
  };

  LLVM_DEBUG(dbgs() << "PrepareTapir: Compute number of parallel iterations\n");
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

//===----------------------------------------------------------------------===//

Value *detail::PrepareTapirLoopGPU::computeOuterLoopTripCount(
    BasicBlock &bb, const TapirLoopInfo &tapirLoop) {
  auto sanityCheck = [](const Loop &loop, ScalarEvolution &se) {
    assert(hasTargetAttr(loop) &&
           "Outer loop must have tapir target attribute");
    assert(loop.getBounds(se) && "Outer loop must have known bounds");
    assert(loop.getCanonicalInductionVariable() &&
           "Outer loop must have canonical induction variable");
  };

  LLVM_DEBUG(dbgs() << "PrepareTapir: Compute number of parallel iterations\n");
  sanityCheck(*tapirLoop.getLoop(), se);

  LLVMContext &ctx = bb.getContext();
  Constant *cnull = ConstantPointerNull::get(PointerType::getUnqual(ctx));

  Loop &loop = *tapirLoop.getLoop();

  Loop::LoopBounds bounds = *loop.getBounds(se);
  PHINode *iv = loop.getCanonicalInductionVariable();
  Value *tc = &bounds.getFinalIVValue();
  Type *ivTy = iv->getType();
  TTID tt = *getTargetAttr(loop);
  Constant *ctt = toConstant(tt, ctx);
  Constant *one = ConstantInt::get(ivTy, 1, /*isSigned=*/false);

  IRBuilder<> builder(&*bb.begin());
  Value *iptRaw = builder.CreateIntrinsic(Intrinsic::kit_gpu_ipt, {ctt, cnull});
  Value *ipt = builder.CreateIntCast(iptRaw, ivTy, /*isSigned=*/false, "ipt");
  Value *iptMinus1 = builder.CreateSub(ipt, one);
  Value *tcPlusIPTMinus1 = builder.CreateAdd(tc, iptMinus1);
  Value *numThreads = builder.CreateUDiv(tcPlusIPTMinus1, ipt);

  return numThreads;
}

//===----------------------------------------------------------------------===//

static bool checkCommon(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                        LoopInfo &li) {
  if (!checkTapirLoopSafeToWrap(tapirLoop, dt, li))
    return false;

  Loop &loop = *tapirLoop.getLoop();

  if (!hasPerfectLevelAttr(loop))
    return emitDiagnostic(loop, DiagID::ErrAttrMissing,
                          getAttrName(LoopAttrKind::PerfectLevel));

  if (isTopLevelTapirLoop(loop) && !hasPerfectDepthAttr(loop))
    return emitDiagnostic(loop, DiagID::ErrAttrMissing,
                          getAttrName(LoopAttrKind::PerfectDepth));

  return true;
}

static bool check(TapirLoopInfo &tapirLoop, DominatorTree &dt, LoopInfo &li) {
  Loop &loop = *tapirLoop.getLoop();
  if (!checkCommon(tapirLoop, dt, li))
    return false;
  else if (hasReductionAttr(loop))
    return detail::checkReductionLoop(tapirLoop, dt, li);
  else
    return detail::checkParallelLoop(tapirLoop, dt, li);
}

static bool prepare(TapirLoopInfo &tapirLoop, DominatorTree &dt, LoopInfo &li,
                    MemorySSA &mssa, ScalarEvolution &se, TaskInfo &ti) {
  if (hasReductionAttr(*tapirLoop.getLoop()))
    return detail::prepareReductionLoop(tapirLoop, dt, li, mssa, se, ti);
  else
    return detail::prepareParallelLoop(tapirLoop, dt, li, mssa, se, ti);
}

static bool prepare(Loop &loop, DominatorTree &dt, LoopInfo &li,
                    MemorySSA &mssa, OptimizationRemarkEmitter &ore,
                    ScalarEvolution &se, TaskInfo &ti) {
  LLVM_DEBUG(dbgs() << "PrepareTapirLoop: Preparing loop '" << getName(loop)
                    << "'\n");

  // Even if the loop is recognized as a tapir loop, if it does not have the
  // correct structure, the transformation that must be performed by this pass
  // will be difficult, if not impossible to perform. Therefore, check this
  // early, and fail immediately. See the comment above the call to check() for
  // a discussion on why we choose to fail instead of producing working, even if
  // slow, code in such cases.
  Task *task = getTaskIfTapirLoopStructure(&loop, &ti);
  if (!task) {
    emitDiagnostic(loop, DiagID::ErrTapirLoopNoTask);
    exitOnError();
  }

  PredicatedScalarEvolution pse(se, loop);
  TapirLoopInfo tapirLoop(&loop, task);

  // Setup the tapir loop object. These must be done before we check if the
  // tapir loop can be transformed, otherwise, the check will definitely fail
  // with spurious errors. We do this early to separate the tasks of setting up
  // the object and checking the loop rather than having the two be
  // interspersed.
  tapirLoop.collectIVs(pse, DEBUG_TYPE, &ore);
  tapirLoop.getOrCreateTripCount(pse, DEBUG_TYPE, &ore);

  // If the tapir loop is such that it cannot be transformed for parallel
  // execution, the entire compilation should fail. At the time of writing this,
  // Kitsune is very much a research prototype, not a production-quality
  // compiler (or even remotely close to it). The goal is not to always produce
  // code that runs, but to push the envelope on the kinds of optimizations that
  // can be performed. Given this objective, it makes more sense to fail if a
  // transformation could not be performed, rather than produce working, albeit
  // slow, code.
  if (!check(tapirLoop, dt, li))
    exitOnError();

  return prepare(tapirLoop, dt, li, mssa, se, ti);
}

PreservedAnalyses PrepareTapirLoopsPass::run(Function &f,
                                             FunctionAnalysisManager &am) {
  DominatorTree &dt = am.getResult<DominatorTreeAnalysis>(f);
  LoopInfo &li = am.getResult<LoopAnalysis>(f);
  MemorySSA &mssa = am.getResult<MemorySSAAnalysis>(f).getMSSA();
  OptimizationRemarkEmitter &ore =
      am.getResult<OptimizationRemarkEmitterAnalysis>(f);
  ScalarEvolution &se = am.getResult<ScalarEvolutionAnalysis>(f);
  TaskInfo &ti = am.getResult<TaskAnalysis>(f);

  bool changed = false;
  SmallVector<Loop *, 4> wl = li.getLoopsInPreorder();
  while (!wl.empty()) {
    // `wl` contains loops in preorder with siblings in forward program order.
    // By popping from the back, we will visit the siblings in reverse program
    // order. This is roughly what we want because it *might* reduce the chances
    // of making a mess of the analysis objects.
    Loop &loop = *wl.pop_back_val();
    if (isTapirLoop(loop) && !hasPreparedAttr(loop))
      changed |= prepare(loop, dt, li, mssa, ore, se, ti);
  }

  if (!changed)
    return PreservedAnalyses::all();
  return getPreservedAnalysesLoop(/*preservedMSSA=*/true);
}

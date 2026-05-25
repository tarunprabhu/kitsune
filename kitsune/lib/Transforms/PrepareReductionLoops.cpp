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
//     int64_t r_mul = 1;
//     parallel_for (int i = 0; i < n; ++i) {
//         r_sum += i;
//         r_mul *= 1;
//     }
//
// Frontends are expected to use the kit.reduce.0 intrinsic to represent the
// loop above
//
//     void sum(int32_t* res, int32_t v) {
//         *res += v;
//     }
//
//     void mul(int64_t* res, int64_t v) {
//         *res *= v;
//     }
//
//     parallel_for (int i = 0; i < n; ++i) {
//         kit.reduce.0(&r_sum, sizeof(r_sum), i, 0, &sum);
//         kit.reduce.0(&r_mul, sizeof(r_mul), i, 1, &mul);
//     }
//
// This pass will transform this into the following for parallel execution on a
// CPU.
//
//     int64_t numPartials = kit.reduce.num.partials(n);
//     int64_t sizePartial = (n + numPartials - 1) / numPartials;
//     int32_t* buf32 = kit.mobile.alloc(numPartials * sizeof(int32_t));
//     int64_t* buf64 = kit.mobile.alloc(numPartials * sizeof(int64_t));
//     kit.mobile.init(buf32, sizeof(int32_t), 0);
//     kit.mobile.init(buf64, sizeof(int64_t), 1);
//     parallel_for (int j = 0; j < numPartials; ++j) {
//         int start = j * sizePartial;
//         int end = std::min(start + sizePartial, n);
//         for (int i = start; i < end; ++i) {
//             kit.reduce.0(&buf32[j], sizeof(r_sum), i, 0, &sum);
//             kit.reduce.0(&buf64[j], sizeof(r_mul), i, 1, &mul);
//         }
//     }
//     kit.reduce.1(&r_sum, buf32, numPartials, 0, &sum);
//     kit.reduce.1(&r_mul, buf64, numPartials, 1, &mul);
//     kit.mobile.free(buf32);
//     kit.mobile.free(buf64);
//
// Here, kit.reduce.num.partials is used to determine the number of partial
// reductions that are to be performed in parallel. An outer parallel loop is
// added to carry these out. Each iteration of the parallel loop will perform a
// sequential reduction. This is followed by calls to kit.reduce.1 which
// computes the final result from the partial reductions.
//
// The code below is the transformation that is carried out for GPU's.
//
//     int64_t numPartials = kit.reduce.num.partials(n);
//     int64_t partialSize = (n + numPartials - 1) / numPartials;
//     int32_t* buf32 = kit.mobile.alloc(numPartials * sizeof(int32_t));
//     int64_t* buf64 = kit.mobile.alloc(numPartials * sizeof(int64_t));
//     kit.mobile.init(buf32, sizeof(int32_t), 0);
//     kit.mobile.init(buf64, sizeof(int64_t), 1);
//     parallel_for (int j = 0; j < numPartials; ++j) {
//         for (int i = j; i < n; i += numPartials) {
//             kit.reduce.0(&buf32[j], sizeof(r_sum), i, 0, &sum);
//             kit.reduce.0(&buf64[j], sizeof(r_mul), i, 1, &mul);
//         }
//     }
//     kit.reduce.1(&r_sum, buf32, numPartials, 0, &sum);
//     kit.reduce.1(&r_mul, buf64, numPartials, 1, &mul);
//     kit.mobile.free(buf32);
//     kit.mobile.free(buf64);
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

#include "kitsune/Transforms/PrepareReductionLoops.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/TapirLoopUtils.h"
#include "kitsune/Core/ValueUtils.h"
#include "kitsune/Support/ErrorHandling.h"
#include "kitsune/Support/TTIDUtils.h"
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

#define DEBUG_TYPE "kit-reductions"

using namespace llvm;

namespace {

/// Information about a single reduction in the tapir reduction loop. A loop
/// may perform more than one such reduction.
struct ReductionInfo {
  CallBase *call = nullptr; ///< Call to the kit_reduce_0 intrinsic

  Value *tt = nullptr;           ///< The TTID of the tapir reduction loop
  Value *dest = nullptr;         ///< The destination for the reduced value
  Value *elemSize = nullptr;     ///< The size (in bytes) of the reduced result
  Value *unit = nullptr;         ///< The unit value for the reduction
  Value *reducer = nullptr;      ///< The reducer function
  SmallVector<Value *, 1> extra; ///< Extra arguments for the reducer function

  Value *partials = nullptr;    ///< The buffer for the partial reductions
  Value *numPartials = nullptr; ///< Number of elements in the partials buffer

  ReductionInfo(CallBase *call)
      : call(call), tt(call->getArgOperand(0)), dest(call->getArgOperand(1)),
        elemSize(call->getArgOperand(2)), unit(call->getArgOperand(4)),
        reducer(call->getArgOperand(5)) {
    for (unsigned i = 6; i < call->arg_size(); ++i)
      extra.push_back(call->getArgOperand(i));
  }
};

/// Base class to transform tapir reduction loops for parallel execution.
/// Specializations of this class will transform the loop for the CPU and GPU
/// respectively.
class PrepareReductionLoop {
private:
  DominatorTree &dt;
  LoopInfo &li;
  ScalarEvolution &se;
  TaskInfo &ti;

  DomTreeUpdater dtu;
  MemorySSAUpdater mssau;

private:
  SmallVector<ReductionInfo, 1> collectReductions(Loop &loop);
  BasicBlock *genOuterPreheaderBlock(Loop &loop);
  BasicBlock *genOuterHeaderBlock(Loop &loop, BasicBlock &outerPh);
  BasicBlock *genOuterReattachBlock(Loop &loop);
  BasicBlock *genOuterLatchBlock(Loop &loop, BasicBlock &outerReattach);
  BasicBlock *genOuterExitBlock(Loop &loop, BasicBlock &outerLatch,
                                BasicBlock &outerHeader);
  BasicBlock *genInnerLoopGuardBlock(Loop &loop, BasicBlock &outerHeader);
  BasicBlock *genInnerLoopEndBlock(Loop &loop, BasicBlock &outerReattach,
                                   BasicBlock &innerGuard);
  void genOuterLoopInsts(BasicBlock &outerLatch, BasicBlock &outerHeader,
                         const TapirLoopInfo &tapirLoop);
  void genOuterLoopMD(BasicBlock &outerLatch, Loop &loop);
  Loop *genOuterLoopObject(Loop &loop, BasicBlock &outerPreheader,
                           BasicBlock &outerHeader, BasicBlock &outerReattach,
                           BasicBlock &outerLatch, BasicBlock &outerExit,
                           BasicBlock &innerGuard, BasicBlock &innerEnd);
  Loop *genOuterLoop(const TapirLoopInfo &tapirLoop);
  BasicBlock *genAllocPartialsBlock(Loop &outerLoop);
  BasicBlock *genReducePartialsBlock(Loop &outerLoop);
  BasicBlock *genFreePartialsBlock(Loop &outerLoop);
  Value *computeNumPartialReductions(BasicBlock &bb, const TapirLoopInfo &loop);
  Value *allocPartialsBuffer(BasicBlock &bb, const ReductionInfo &info);
  void initPartialsBuffer(BasicBlock &bb, Loop &loop,
                          const ReductionInfo &info);
  void reduceIntoPartialsBuffer(Loop &outerLoop, Loop &loop,
                                const ReductionInfo &info);
  void genFinalReduction(BasicBlock &bb, const ReductionInfo &info);
  void freePartialsBuffer(BasicBlock &bb, const ReductionInfo &info);
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
      : dt(dt), li(li), se(se), ti(ti),
        dtu(dt, DomTreeUpdater::UpdateStrategy::Eager), mssau(&mssa) {}

  bool run(TapirLoopInfo &tapirLoop);
};

} // namespace

// Mark the loop as having been prepared. \p hasChanged must be true if the loop
// has been changed by the caller in any way. Otherwise, it should be false.
// Returns \p hasChanged.
static bool annotateLoopAsPrepared(Loop &loop, bool hasChanged) {
  addReductionPreparedAttr(loop);
  return hasChanged;
}

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

// Generate a block for the preheader of the outer loop. This will be inserted
// after the preheader of \p loop. The figure below is an approximation of the
// CFG after this function returns.
//
//     LoopPreheader
//     OuterPreheader
//     LoopPreheaderNew
//         LoopHeader
//         <LoopBlocks>
//         LoopLatch
//     LoopExit
//
// When we add the outer loop header and parallelize it, that header will
// terminate in a detach instruction, which has two successors. The original
// loop would then no longer be in loop-simplify form since it would not have a
// preheader. To ensure that the original loop remains in loop-simplify form,
// add an empty preheader for the original loop.
//
// By adding the outer loop preheader *after* the original loop preheader, we
// ensure that any code, particularly any syncregions and allocs created in that
// preheader will be created before the outer loop. Otherwise, these will end
// up inside what will eventually become a parallel loop - leading to avoidable
// complications.
BasicBlock *PrepareReductionLoop::genOuterPreheaderBlock(Loop &loop) {
  LLVM_DEBUG(dbgs() << "PrepareReduction:   Create preheader block\n");

  BasicBlock *ph = loop.getLoopPreheader();
  BasicBlock *bb = SplitBlock(ph, ph->getTerminator(), &dtu, &li, &mssau,
                              "prduc.ph", /*Before=*/false);

  // At this point, we have created the outer loop preheader *after* the
  // original loop preheader. Split this to get a new preheader for the inner
  // loop.
  (void)SplitBlock(bb, bb->getTerminator(), &dtu, &li, &mssau,
                   "prduc.inner.ph.new", /*Before=*/false);

  return bb;
}

// Generate a block for the header of the outer loop. This will be inserted
// before the preheader of \p loop. The figure below is an approximation of the
// CFG after this function returns.
//
//     LoopPreheader
//     OuterPreheader
//         OuterHeader
//         LoopPreheaderNew
//             LoopHeader
//             <LoopBlocks>
//             LoopLatch
//         LoopExit
//
// NOTE: In the figure above, we have indented the blocks from \p loop for
// clarity. However, the LoopInfo object will *NOT* be modified here, so it will
// not know about the outer loop.
//
// This function will generate a PHINode in the generated header block that will
// (eventually) be the primary induction variable of the loop. That node will
// only have a single incoming value of 0 from the outer loop preheader,
// \p outerPh.
BasicBlock *PrepareReductionLoop::genOuterHeaderBlock(Loop &loop,
                                                      BasicBlock &outerPh) {
  LLVM_DEBUG(dbgs() << "PrepareReduction:   Create header block\n");

  BasicBlock *bb = SplitBlock(&outerPh, outerPh.getTerminator(), &dtu, &li,
                              &mssau, "prduc.header", /*Before=*/false);

  // Add a phi node to this header. This will be the primary induction variable
  // of the outer loop.
  LLVMContext &ctx = getContext(loop);
  Type *i64 = Type::getInt64Ty(ctx);
  PHINode *iv =
      PHINode::Create(i64, /*NumReserved=*/2, "prduc.iv", bb->begin());
  Constant *zero = ConstantInt::getSigned(i64, 0);
  iv->addIncoming(zero, &outerPh);

  return bb;
}

// Generate a basic block that will be the eventually contain a single reattach
// instruction for the outer loop body. This will be inserted after the sole
// exit block of \p loop. The figure below is an approximation of the CFG after
// this function returns.
//
//     LoopPreheader
//     OuterPreheader
//         OuterHeader
//         LoopPreheaderNew
//             LoopHeader
//             <LoopBlocks>
//             LoopLatch
//         LoopExitNew
//         OuterReattach
//     LoopExit
//
// In the figure above, the original LoopExit block has been split. The
// instructions that were originally in the exit block of \p loop will be
// in LoopExit. The LoopExit block will often contain a sync instruction. Moving
// the instructions this way ensures that the sync will appear *after* the outer
// loop. LoopExitNew will be empty.
BasicBlock *PrepareReductionLoop::genOuterReattachBlock(Loop &loop) {
  LLVM_DEBUG(dbgs() << "PrepareReduction:   Create reattach block\n");

  // Split the loop exit block. This results in two blocks. The first, which is
  // returned from SplitBlock(), will be an empty block corresponding to
  // LoopExitNew in the figure above. LoopExit will remain essentially
  // unchanged.
  BasicBlock *loopExit = loop.getExitBlock();
  (void)SplitBlock(loopExit, loopExit->begin(), &dtu, &li, &mssau,
                   "prduc.inner.exit.new", /*Before=*/true);

  // Split the loop exit block again. SplitBlock() will once again return an
  // empty block while LoopExit will remain effectively unchanged.
  BasicBlock *bb = SplitBlock(loopExit, loopExit->begin(), &dtu, &li, &mssau,
                              "prduc.reattach", /*Before=*/true);

  return bb;
}

// Generate a basic block that will be the outer loop latch. This will be
// inserted after the sole exit block of \p loop. The figure below is an
// approximation of the CFG after this function returns.
//
//     LoopPreheader
//     OuterPreheader
//         OuterHeader
//         LoopPreheaderNew
//             LoopHeader
//             <LoopBlocks>
//             LoopLatch
//         LoopExitNew
//         OuterReattach
//         OuterLatch
//     LoopExit
//
BasicBlock *
PrepareReductionLoop::genOuterLatchBlock(Loop &loop,
                                         BasicBlock &outerReattach) {
  LLVM_DEBUG(dbgs() << "PrepareReduction:   Create latch block\n");

  BasicBlock *bb =
      SplitBlock(&outerReattach, outerReattach.getTerminator(), &dtu, &li,
                 &mssau, "prduc.latch", /*Before=*/false);

  // At this point, the latch branches unconditionally to LoopExit. It will be
  // replaced with an appropriate conditional branch and backedge when the outer
  // outer loop exit block is created.
  return bb;
}

// Generate a basic block that will be the sole outer loop exit block. This will
// be inserted after \p outerLatch. The figure below shows what the CFG is
// expected to be after this function returns.
//
//     LoopPreheader
//     OuterPreheader
//         OuterHeader
//         LoopPreheaderNew
//             LoopHeader
//             <LoopBlocks>
//             LoopLatch
//         LoopExitNew
//         OuterReattach
//         OuterLatch
//     OuterExit
//     LoopExit
//
// The terminator of \p outerLatch will be updated to be a conditional branch
// that goes to either \p outerHeader, or the generated exit block. However,
// the condition of the branch will not be valid.
BasicBlock *PrepareReductionLoop::genOuterExitBlock(Loop &loop,
                                                    BasicBlock &outerLatch,
                                                    BasicBlock &outerHeader) {
  LLVM_DEBUG(dbgs() << "PrepareReduction:   Create exit block\n");

  BasicBlock *bb = SplitBlock(&outerLatch, outerLatch.getTerminator(), &dtu,
                              &li, &mssau, "prduc.exit", /*Before=*/false);

  // Update the terminator of the outer loop latch. We don't update the
  // induction variable of the outer loop since we don't yet have an appropriate
  // value that will be sent along the backedge. However, we don't yet have a
  // proper condition to check for in the branch, so just use a placeholder.
  LLVMContext &ctx = getContext(loop);
  Constant *cond = ConstantInt::getTrue(ctx);
  BranchInst *br = BranchInst::Create(bb, &outerHeader, cond);
  ReplaceInstWithInst(outerLatch.getTerminator(), br);

  return bb;
}

// Generate a basic block that will be the guard for the inner loop. The
// original inner loop guard will be a predecessor of the loop preheader.
// However, once the loop is transformed, the inner loop bounds will change, so,
// a guard may well be necessary. The figure below shows what the CFG is
// expected to be after this function returns.
//
//     LoopPreheader
//     OuterPreheader
//         OuterHeader
//         LoopGuardNew
//         LoopPreheaderNew
//             LoopHeader
//             <LoopBlocks>
//             LoopLatch
//         LoopExitNew
//         OuterReattach
//         OuterLatch
//     OuterExit
//     LoopExit
//
// The terminator of the new guard will be unconditional and will branch to the
// preheader of the inner loop.
BasicBlock *
PrepareReductionLoop::genInnerLoopGuardBlock(Loop &loop,
                                             BasicBlock &outerHeader) {
  LLVM_DEBUG(dbgs() << "PrepareReduction:   Create loop guard block\n");

  Instruction *term = outerHeader.getTerminator();
  BasicBlock *bb = SplitBlock(&outerHeader, term, &dtu, &li, &mssau,
                              "prduc.inner.guard.new", /*Before=*/false);

  return bb;
}

// Generate a basic block that will be the end block for the inner loop. This
// will be the destination for the inner loop guard if that loop is to be
// bypassed. The figure below shows what the CFG is expected to be after this
// function returns.
//
//     LoopPreheader
//     OuterPreheader
//         OuterHeader
//         LoopGuardNew
//         LoopPreheaderNew
//             LoopHeader
//             <LoopBlocks>
//             LoopLatch
//         LoopExitNew
//         LoopEndNew
//         OuterReattach
//         OuterLatch
//     OuterExit
//     LoopExit
//
// A conditional branch will be inserted between the \p innerGuard and the new
// inner loop end block. The conditional of that branch will not be valid and
// will be updated later in this transformation.
BasicBlock *PrepareReductionLoop::genInnerLoopEndBlock(
    Loop &loop, BasicBlock &outerReattach, BasicBlock &innerGuard) {
  auto sanityCheck = [](Loop &loop) {
    assert(loop.getLoopPreheader() && "Inner loop must have a preheader");
  };

  LLVM_DEBUG(dbgs() << "PrepareReduction:   Create loop end block\n");
  sanityCheck(loop);

  LLVMContext &ctx = getContext(loop);
  Instruction *term = outerReattach.getTerminator();
  BasicBlock *innerEnd = SplitBlock(&outerReattach, term, &dtu, &li, &mssau,
                                    "prduc.inner.end.new", /*Before=*/true);

  // Now that we have the end block, replace the terminator of the inner loop
  // guard with a conditional branch.
  BasicBlock *ph = loop.getLoopPreheader();
  Constant *cond = ConstantInt::getFalse(ctx);
  BranchInst *br = BranchInst::Create(innerEnd, ph, cond);

  ReplaceInstWithInst(innerGuard.getTerminator(), br);

  // Update analyses
  dt.insertEdge(&innerGuard, innerEnd);

#ifndef NDEBUG
  dt.verify();
#endif // NDEBUG

  return innerEnd;
}

// Generate the correct instructions in \outerLatch such that it will iterate
// correctly. This will update the terminator of the loop latch, as well as the
// loop induction variable in the outer loop header, \p outerHeader. \p loop is
// the tapir loop corresponding to the loop being transformed.
void PrepareReductionLoop::genOuterLoopInsts(BasicBlock &outerLatch,
                                             BasicBlock &outerHeader,
                                             const TapirLoopInfo &tapirLoop) {
  auto sanityCheck = [](BasicBlock &outerLatch, BasicBlock &outerHeader) {
    BranchInst *outerBr = dyn_cast<BranchInst>(outerLatch.getTerminator());
    assert(outerBr &&
           "Terminator of outer loop latch must be a branch instruction");
    assert(outerBr->getNumSuccessors() == 2 &&
           "Terminator of outer loop latch must be a conditional branch");

    PHINode *outerIV = dyn_cast<PHINode>(outerHeader.begin());
    assert(outerIV &&
           "First instruction of outer loop header must be a phi node");
  };

  LLVM_DEBUG(dbgs() << "PrepareReduction:   Add latch instructions\n");
  sanityCheck(outerLatch, outerHeader);

  BranchInst *outerBr = cast<BranchInst>(outerLatch.getTerminator());
  IRBuilder<> builder(outerBr);

  PHINode *outerIV = cast<PHINode>(outerHeader.begin());
  Constant *one = ConstantInt::getSigned(outerIV->getType(), 1);
  Value *outerInc = builder.CreateAdd(outerIV, one, "prduc.iv.inc");

  Instruction *innerInc = getPrimaryIVInc(tapirLoop);
  cast<Instruction>(outerInc)->copyIRFlags(innerInc);

  outerIV->addIncoming(outerInc, &outerLatch);

  Value *tc = tapirLoop.getTripCount();
  Value *outerCmp = builder.CreateICmpEQ(outerInc, tc, "prduc.iv.cmp");
  outerBr->setCondition(outerCmp);
}

// Generate loop metadata for the outer loop. This will be a clone of the
// metadata of the loop, \p loop being transformed.
void PrepareReductionLoop::genOuterLoopMD(BasicBlock &outerLatch, Loop &loop) {
  LLVM_DEBUG(dbgs() << "PrepareReduction:   Create loop metadata\n");

  MDNode *loopMD = loop.getLoopID();
  assert(loopMD && "Loop being transformed does not have metadata");

  TempMDNode tempMD = loopMD->clone();
  MDNode *md = MDNode::replaceWithDistinct(std::move(tempMD));
  md->replaceOperandWith(0, md);

  outerLatch.getTerminator()->setMetadata(LLVMContext::MD_loop, md);
}

// Generate a loop object for the outer loop with the given blocks. The
// structure of the loop will be as follows:
//
//     LoopPreheader
//     OuterPreheader
//         OuterHeader
//         LoopGuardNew
//         LoopPreheaderNew
//             LoopHeader
//             <LoopBlocks>
//             LoopLatch
//         LoopExitNew
//         LoopEndNew
//         OuterReattach
//         OuterLatch
//     OuterExit
//     LoopExit
//
// LoopPreheader, LoopHeader, <LoopBlocks>, LoopLatch, and LoopExit in the
// figure above are from \p loop that will now become the sole child of the
// outer loop object that will be created here. If the original depth of \p loop
// was `d`, it will be `d+1` after this function is returned while the newly
// created outer loop object will be at depth `d`. LoopPreheaderNew and
// LoopExitNew are empty blocks that were created when the outer loop blocks
// were created.
//
// This will update the analysis objects in this class, so they should be safe
// to use after this returns.
Loop *PrepareReductionLoop::genOuterLoopObject(
    Loop &loop, BasicBlock &outerPh, BasicBlock &outerHeader,
    BasicBlock &outerReattach, BasicBlock &outerLatch, BasicBlock &outerExit,
    BasicBlock &innerGuard, BasicBlock &innerEnd) {
  auto sanityCheck = [](Loop &loop, BasicBlock &outerHeader,
                        BasicBlock &outerReattach, BasicBlock &outerLatch,
                        BasicBlock &outerExit, BasicBlock &innerGuard,
                        BasicBlock &innerEnd) {
    BasicBlock *ph = loop.getLoopPreheader();
    BasicBlock *exit = loop.getExitBlock();

    assert(ph && "Inner loop must have a preheader");
    assert(exit && "Inner loop must have a unique exit block");

    assert(succ_size(&outerHeader) == 1 &&
           "Outer loop header must have a single successor");
    assert(*succ_begin(&outerHeader) == &innerGuard &&
           "Successor of outer loop header must be the inner loop guard");

    BranchInst *br = dyn_cast<BranchInst>(innerGuard.getTerminator());
    assert(br && "Terminator of inner loop guard must be a branch instruction");
    assert(br->getNumSuccessors() == 2 &&
           "Inner loop guard must have exactly two successors");
    assert(br->getSuccessor(0) == &innerEnd &&
           "First sucessor of inner loop guard must be loop end block");
    assert(br->getSuccessor(1) == ph &&
           "Second successor of inner loop guard must be the loop preheader");

    assert(pred_size(&innerEnd) == 2 &&
           "Inner loop end block must have exactly two predecessors");

    assert(succ_size(exit) == 1 &&
           "Inner loop exit block must have exactly one successor");
    assert(*succ_begin(exit) == &innerEnd &&
           "Successor of inner loop exit block must be the loop end block");

    assert(succ_size(&innerEnd) == 1 &&
           "Inner loop end block must have exactly one successor");
    assert(*succ_begin(&innerEnd) == &outerReattach &&
           "Successor of inner loop end block must be the outer loop reattach "
           "block");

    assert(pred_size(&outerReattach) == 1 &&
           "Outer loop reattach block must have a single predecessor");
    assert(*pred_begin(&outerReattach) == &innerEnd &&
           "Predecessor of outer loop reattach block must be inner loop exit");
    assert(succ_size(&outerReattach) == 1 &&
           "Outer loop reattach block must have a single successor");
    assert(*succ_begin(&outerReattach) == &outerLatch &&
           "Successor of outer loop reattach block must be outer loop latch");

    assert(pred_size(&outerLatch) == 1 &&
           "Outer loop latch must have a single predecessor");
    assert(*pred_begin(&outerLatch) == &outerReattach &&
           "Predecessor of outer loop latch must be outer loop reattach block");

    assert(pred_size(&outerExit) == 1 &&
           "Outer loop exit must have a single predecessor");
    assert(*pred_begin(&outerExit) == &outerLatch &&
           "Predecessor of outer loop exit must be outer loop latch");
  };

  LLVM_DEBUG(dbgs() << "PrepareReduction:   Create loop object\n");
  sanityCheck(loop, outerHeader, outerReattach, outerLatch, outerExit,
              innerGuard, innerEnd);

  Loop *outerLoop = li.AllocateLoop();
  if (Loop *parentLoop = loop.getParentLoop())
    parentLoop->replaceChildLoopWith(&loop, outerLoop);
  else
    li.changeTopLevelLoop(&loop, outerLoop);
  outerLoop->addChildLoop(&loop);

  BasicBlock *ph = loop.getLoopPreheader();
  BasicBlock *exit = loop.getExitBlock();

  // Add blocks to the outer loop. We add them in roughly the same order as the
  // figure above just to keep things somewhat organized. The blocks in the
  // inner loop must also be added to the new outer loop. These don't have to be
  // added to the parents of the outer loop (if any) since they should already
  // be present there, so we just use addBlockEntry().
  outerLoop->addBasicBlockToLoop(&outerHeader, li);
  outerLoop->addBasicBlockToLoop(&innerGuard, li);
  outerLoop->addBasicBlockToLoop(ph, li);
  for (BasicBlock *bb : loop.getBlocks())
    outerLoop->addBlockEntry(bb);
  outerLoop->addBasicBlockToLoop(exit, li);
  outerLoop->addBasicBlockToLoop(&innerEnd, li);
  outerLoop->addBasicBlockToLoop(&outerReattach, li);
  outerLoop->addBasicBlockToLoop(&outerLatch, li);

  assert(loop.isLoopSimplifyForm() &&
         "Inner loop must be maintained in loop-simplify form");
  assert(outerLoop->isLoopSimplifyForm() &&
         "Outer loop must be generated in loop-simplify form");

#ifndef NDEBUG
  dt.verify();
  li.verify(dt);
  loop.verifyLoop();
  outerLoop->verifyLoop();
#endif // NDEBUG

  return outerLoop;
}

// The current loop structure is as shown below. The LoopPreheader and
// LoopExit blocks are not part of the loop, so they are indented at a
// different level.
//
//     LoopPreheader
//         LoopHeader
//         <LoopBlocks>
//         LoopLatch
//     LoopExit
//
// This has to be transformed to the following:
//
//     LoopPreheader
//     OuterPreheader
//         OuterHeader
//         LoopGuardNew
//         LoopPreheaderNew
//             LoopHeader
//             <LoopBlocks>
//             LoopLatch
//         LoopExitNew
//         LoopEndNew
//         OuterReattach
//         OuterLatch
//     OuterExit
//     LoopExit
//
Loop *PrepareReductionLoop::genOuterLoop(const TapirLoopInfo &tapirLoop) {
  LLVM_DEBUG(dbgs() << "PrepareReduction: Generate outer loop\n");

  Loop &loop = *tapirLoop.getLoop();

  // We first create the core blocks required by the outer loop. At no stage
  // during the creation of these can the CFG be assumed to be "correct" i.e.
  // blocks may be orphaned, induction variables may not have the correct set of
  // incoming values and blocks, conditional branch instructions may have
  // invalid conditions etc. The CFG will be kept "somewhat sane", but that's
  // about it.
  BasicBlock *outerPreheader = genOuterPreheaderBlock(loop);
  BasicBlock *outerHeader = genOuterHeaderBlock(loop, *outerPreheader);
  BasicBlock *outerReattach = genOuterReattachBlock(loop);
  BasicBlock *outerLatch = genOuterLatchBlock(loop, *outerReattach);
  BasicBlock *outerExit = genOuterExitBlock(loop, *outerLatch, *outerHeader);

  // Add additional basic blocks to guard the inner loop.
  BasicBlock *innerGuard = genInnerLoopGuardBlock(loop, *outerHeader);
  BasicBlock *innerEnd =
      genInnerLoopEndBlock(loop, *outerReattach, *innerGuard);

  // We have now created the "core" basic blocks for the outer loop and the CFG
  // is technically correct i.e. the outer loop contains a preheader with an
  // unconditional branch to the header, the loop latch contains a backedge to
  // the header and the unique loop exit block and the header dominates that
  // loop exit block.
  //
  // However, the induction variable does not contain the correct number of
  // incoming values and blocks, it is not even updated, and the conditional
  // expression in the terminator of the loop latch is wrong.
  //
  // These are fixed, though with yet more placeholders - specifically, we
  // use the trip count of the inner loop as the trip count of the outer loop
  // and 1 as the step. The trip counts and steps for both the outer and inner
  // loops have to be changed eventually, so we might as well keep things
  // "consistent" here.
  //
  // And yes, we won't pretend that this makes any sense.
  genOuterLoopInsts(*outerLatch, *outerHeader, tapirLoop);

  // Generate appropriate loop metadata for the outer loop. This will just be a
  // clone of the metadata of the original loop.
  genOuterLoopMD(*outerLatch, loop);

  // Generate a loop object for the outer loop with the given blocks. This will
  // also update the LoopInfo object, so it should be safe to use it after
  // this.
  Loop *outerLoop =
      genOuterLoopObject(loop, *outerPreheader, *outerHeader, *outerReattach,
                         *outerLatch, *outerExit, *innerGuard, *innerEnd);

  LLVM_DEBUG(dbgs() << "PrepareReduction: Done generating outer loop\n");
  LLVM_DEBUG(dbgs() << "PrepareReduction:\n" << *outerLoop);

  return outerLoop;
}

// Generate a basic block where the buffers that will contain the partial
// reductions are allocated. The figure below shows what the CFG is expected to
// be after this function returns.
//
//     LoopPreheader
//     PartialsAlloc
//     OuterPreheader
//         OuterHeader
//         LoopGuardNew
//         LoopPreheaderNew
//             LoopHeader
//             <LoopBlocks>
//             LoopLatch
//         LoopExitNew
//         LoopEndNew
//         OuterReattach
//         OuterLatch
//     OuterExit
//     LoopExit
//
// NOTE: We could just add the allocations to the preheader of the outer loop.
// However, since we will go on to create a dedicated block where these buffers
// will be freed, we create a block where they will be allocated purely for
// symmetry. SimplifyCFG will be called after this pass anyway, at which point,
// these blocks will likely get merged anyway.
BasicBlock *PrepareReductionLoop::genAllocPartialsBlock(Loop &outerLoop) {
  LLVM_DEBUG(dbgs() << "PrepareReduction: Generate allocate partials block\n");

  BasicBlock *outerPh = outerLoop.getLoopPreheader();
  assert(outerPh && "Outer loop must have a preheader");

  BasicBlock *bb = SplitBlock(outerPh, outerPh->begin(), &dtu, &li, &mssau,
                              "prduc.partial.alloc", /*Before=*/true);
  return bb;
}

// Generate a basic block where the partial reductions are reduced to the final
// result. In the current implementation, we simply insert calls to the
// `@llvm.kit.reduce.1d` intrinsic, so everything can be added to a single
// block (a later pass will lower the intrinsic as required). The figure below
// shows what the CFG is expected to be after this function returns.
//
//     LoopPreheader
//     PartialsAlloc
//     OuterPreheader
//         OuterHeader
//         LoopGuardNew
//         LoopPreheaderNew
//             LoopHeader
//             <LoopBlocks>
//             LoopLatch
//         LoopExitNew
//         LoopEndNew
//         OuterReattach
//         OuterLatch
//     OuterExit
//     PartialsReduce
//     PartialsFree
//     LoopExit
//
BasicBlock *PrepareReductionLoop::genReducePartialsBlock(Loop &outerLoop) {
  LLVM_DEBUG(dbgs() << "PrepareReduction: Generate reduce partials block\n");

  BasicBlock *outerExit = outerLoop.getExitBlock();
  assert(outerExit && "Outer loop must have a unique exit block");

  BasicBlock *bb = SplitBlock(outerExit, outerExit->getTerminator(), &dtu, &li,
                              &mssau, "prduc.partial.reduce", /*Before=*/false);
  return bb;
}

// Generate a basic block where the buffers containing the partial reductions
// are freed. The figure below shows what the CFG is expected to be after this
// function returns.
//
//     LoopPreheader
//     PartialsAlloc
//     OuterPreheader
//         OuterHeader
//         LoopGuardNew
//         LoopPreheaderNew
//             LoopHeader
//             <LoopBlocks>
//             LoopLatch
//         LoopExitNew
//         LoopEndNew
//         OuterReattach
//         OuterLatch
//     OuterExit
//     PartialsFree
//     LoopExit
//
BasicBlock *PrepareReductionLoop::genFreePartialsBlock(Loop &outerLoop) {
  LLVM_DEBUG(dbgs() << "PrepareReduction: Generate free partials block\n");

  BasicBlock *outerExit = outerLoop.getExitBlock();
  assert(outerExit && "Outer loop must have a unique exit block");

  BasicBlock *bb = SplitBlock(outerExit, outerExit->getTerminator(), &dtu, &li,
                              &mssau, "prduc.partial.free", /*Before=*/false);

  return bb;
}

// Insert code to calculate the number of partial reductions to use. This
// simply inserts a call to Kitsune's kit.reduce.num.partials intrinsic that
// performs the actual. The intrinsic will be lowered in a later pass.
//
//      prduc.numPartials = call @llvm.kit.reduce.num.partials(i64 %n)
//
// Here %n is the trip count of the tapir reduction loop being transformed,
// \p tapirLoop
Value *PrepareReductionLoop::computeNumPartialReductions(
    BasicBlock &bb, const TapirLoopInfo &tapirLoop) {
  auto sanityCheck = [](const TapirLoopInfo &tapirLoop) {
    assert(tapirLoop.getTripCount() &&
           "Expected finite trip count in tapir reduction loop");
    assert(hasTargetAttr(*tapirLoop.getLoop()) &&
           "Outer loop must have tapir target attribute");
  };

  LLVM_DEBUG(
      dbgs() << "PrepareReduction: Compute number of partial reductions\n");
  sanityCheck(tapirLoop);

  LLVMContext &ctx = bb.getContext();
  Type *i64 = Type::getInt64Ty(ctx);

  Loop &loop = *tapirLoop.getLoop();
  TTID tt = *getTargetAttr(loop);
  Constant *ctt = toConstant(tt, ctx);
  Value *tc = tapirLoop.getTripCount();

  IRBuilder<> builder(&*bb.begin());
  Value *tc64 = builder.CreateIntCast(tc, i64, /*isSigned=*/true);
  Value *numPartials =
      builder.CreateIntrinsic(Intrinsic::kit_reduce_num_partials, {ctt, tc64},
                              /*FMFSource=*/{}, "prduc.num.partials");

  return numPartials;
}

// Generate code to allocate the buffer where the partial reductions will be
// stored. This is simply inserts a call to Kitsune's kit.mobile.alloc
// intrinsic. The returned buffer may not have been initialized. In any case,
// it must be initialized with the unit value, but that will be done elsewhere.
Value *PrepareReductionLoop::allocPartialsBuffer(BasicBlock &bb,
                                                 const ReductionInfo &info) {
  LLVM_DEBUG(dbgs() << "PrepareReduction: Allocate buffers for partials\n");

  assert(info.numPartials && "Number of partial reductions must be set");

  LLVMContext &ctx = bb.getContext();
  Type *i64 = Type::getInt64Ty(ctx);

  IRBuilder builder(bb.getTerminator());

  Value *n64 = builder.CreateIntCast(info.numPartials, i64, /*isSigned=*/true);
  Value *sz64 = builder.CreateIntCast(info.elemSize, i64, /*isSigned=*/true);
  Value *bytes = builder.CreateNUWMul(sz64, n64, "prduc.bytes");
  Value *buf = builder.CreateIntrinsic(Intrinsic::kit_mobile_alloc, bytes,
                                       /*FMFSource=*/{}, "prduc.reds");

  return buf;
}

void PrepareReductionLoop::initPartialsBuffer(BasicBlock &bb, Loop &loop,
                                              const ReductionInfo &info) {
  LLVM_DEBUG(dbgs() << "PrepareReduction: Initialize partials buffer\n");

  LLVMContext &ctx = bb.getContext();
  Constant *ctt = toConstant(*getTargetAttr(loop), ctx);
  Value *partials = info.partials;
  Value *numPartials = info.numPartials;
  Value *unit = info.unit;

  IRBuilder<> builder(bb.getTerminator());

  builder.CreateIntrinsic(Intrinsic::kit_mobile_init, {unit->getType()},
                          {ctt, partials, numPartials, unit});
}

// Consider a reduction loop of the form:
//
//     parallel_for (int i = 0; i < n; ++i)
//       a += ...
//
// After the loops have been transformed, and a partials buffer allocated, the
// the actual reductions should be modified use that buffer. The reduction in
// the loop above should be transformed as shown below, where j is the
// induction variable of the outer loop.
//
//       partials[j] += ...
//
// Since the reduction will be represented by a call to the `kit_reduce_0`
// intrinsic, this function will change the first operand of that instruction.
void PrepareReductionLoop::reduceIntoPartialsBuffer(Loop &outerLoop,
                                                    Loop &innerLoop,
                                                    const ReductionInfo &info) {
  LLVM_DEBUG(dbgs() << "PrepareReduction: Reduce into partials buffer\n");

  assert(outerLoop.getInductionVariable(se) &&
         "Outer loop must have an induction variable");

  CallBase *call = info.call;
  Value *unit = info.unit;
  Value *partials = info.partials;

  // The unit value of the reduction will have the same type as the elements
  // being reduced, so we can use that as the element type of the GEP.
  Type *destTy = call->getArgOperand(1)->getType();
  Type *elemTy = unit->getType();
  Value *idx = outerLoop.getInductionVariable(se);

  IRBuilder<> builder(call);

  // Since we have generated the code, we know that the access will be in
  // bounds. If it is not, we have a much bigger problem.
  Value *addr =
      builder.CreateInBoundsGEP(elemTy, partials, {idx}, "prduc.partialsidx");
  Value *addrCast = builder.CreatePointerBitCastOrAddrSpaceCast(addr, destTy);
  call->setArgOperand(1, addrCast);
}

// Generate code to perform the final reduction over the partial reductions.
// This simply inserts a call to Kitsune's `kit.reduce.1` intrinsic that
// performs a reduction over a contiguous 1D array. The lowering of this
// intrinsic will be handled in a later pass. The call is added at the end of
// the basic block \p bb.
void PrepareReductionLoop::genFinalReduction(BasicBlock &bb,
                                             const ReductionInfo &info) {
  LLVM_DEBUG(dbgs() << "PrepareReduction: Generate final reductions\n");

  IRBuilder<> builder(bb.getTerminator());
  SmallVector<Value *, 8> args = {
      info.tt,          info.dest, info.elemSize, info.partials,
      info.numPartials, info.unit, info.reducer};
  for (Value *v : info.extra)
    args.push_back(v);

  Type *elemTy = info.unit->getType();

  (void)builder.CreateIntrinsic(Intrinsic::kit_reduce_1, {elemTy}, args);
}

// Generate code to free the buffer containing the partial reductions. This
// will simply insert a call to Kitsune's `kit.mobile.free` intrinsic.
void PrepareReductionLoop::freePartialsBuffer(BasicBlock &bb,
                                              const ReductionInfo &reduction) {
  LLVM_DEBUG(dbgs() << "PrepareReduction: Free buffers for partials\n");

  IRBuilder builder(bb.getTerminator());
  (void)builder.CreateIntrinsic(Intrinsic::kit_mobile_free, reduction.partials);
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
  unsigned idx = cmp->getOperand(0) == inc ? 1 : 0;

  cmp->setOperand(idx, numPartials);

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

    assert(pred_size(innerLoop.getLoopPreheader()) == 1 &&
           "Inner loop preheader must have a single predecessor");
    assert(succ_size(innerLoop.getExitBlock()) == 1 &&
           "Inner loop exit block must have a single successor");

    assert(innerLoop.getLoopGuardBranch() && "Inner loop must have a guard");
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

  unsigned idx = cmp->getOperand(1) == tc ? 1 : 0;

  iv->setIncomingValueForBlock(ph, newStart);
  cmp->setOperand(idx, newEnd);

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

  unsigned idx = innerStep->getOperand(0) == innerIV ? 1 : 0;

  innerIV->setIncomingValueForBlock(innerPh, outerIV);
  innerStep->setOperand(idx, numPartials);

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
  auto sanityCheck = [](Loop &loop, ScalarEvolution &se) {
    BranchInst *br = loop.getLoopGuardBranch();
    assert(br && "Inner loop must have a guard block");

    assert(isFalse(br->getCondition()) &&
           "Temporary condition of inner loop guard branch must be `false`");

    assert(loop.getBounds(se).has_value() &&
           "Inner loop must have finite bounds");
  };

  LLVM_DEBUG(dbgs() << "PrepareReduction: Update inner loop guard condition\n");
  sanityCheck(loop, se);

  Loop::LoopBounds bounds = *loop.getBounds(se);
  Value *start = &bounds.getInitialIVValue();
  Value *end = &bounds.getFinalIVValue();

  BranchInst *br = loop.getLoopGuardBranch();
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
    BasicBlock *innerExit = loop.getExitBlock();
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

  auto getSyncRegion = [](Loop &loop) -> Value * {
    for (BasicBlock *bb : getBlocksNotInSubLoops(loop))
      for (Instruction &inst : *bb)
        if (auto *detach = dyn_cast<DetachInst>(&inst))
          return detach->getSyncRegion();
    llvm_unreachable("Could not get syncregion for tapir reduction loop");
  };

  LLVM_DEBUG(dbgs() << "PrepareReduction: Parallelize outer loop\n");
  sanityCheck(outerLoop, loop, ti);

  Value *syncRegion = getSyncRegion(loop);

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
  assert(getTaskIfTapirLoop(&loop, &ti) &&
         "Inner loop not recognized as tapir loop after outer loop was "
         "parallalized");
#endif // NDEBUG
}

void PrepareReductionLoop::serializeInnerLoop(Loop &loop) {
  assert(loop.isLoopSimplifyForm() &&
         "Inner loop must be in loop-simplify form");

  Task *task = getTaskIfTapirLoop(&loop, &ti);
  serializeTapirLoop(loop, *task, &dt, &ti);

  // serializeTapirLoop will already have updated the analyses that were
  // invalidated by this transformation

  // FIXME: Recalculating everything in TaskInfo is wasteful, but it is not
  // clear that there is a better way to do it currently.
  Function *f = getFunction(loop);
  ti.recalculate(*f, dt);

#ifndef DEBUG
  ti.verify(dt);
  assert(!getTaskIfTapirLoop(&loop, &ti) &&
         "Inner loop recognized as tapir loop after serialization");
#endif // NDEBUG
}

// Transform a tapir reduction loop. Add the tapir.loop.reduction.prepared
// attribute to the loop and return true if anything other than this attribute
// was changed. For example, if the loop does not perform any actual reductions,
// the attribute will be added to the loop, but no other transformations will be
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
    assert(!hasReductionPreparedAttr(loop) &&
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
  if (reductions.empty())
    return annotateLoopAsPrepared(loop, /*hasChanged=*/false);

  // Generate the outer loop including all blocks. The analysis objects will
  // have been updated.
  Loop *outerLoop = genOuterLoop(tapirLoop);

  // Generate additional blocks to add other code that will be needed. These
  // will be added on either side of the outer loop.
  BasicBlock *bbAllocPartials = genAllocPartialsBlock(*outerLoop);
  BasicBlock *bbFreePartials = genFreePartialsBlock(*outerLoop);
  BasicBlock *bbReducePartials = genReducePartialsBlock(*outerLoop);

  // Don't make any changes to the loop trip counts just yet. While the
  // transformations to the reduction loop should not be affected by any change
  // to the trip count, it may be safer to leave them as canonical until we have
  // finished all the other transformations.

  // This inserts code for the allocation, initialization, use and deallocation
  // of the partial reduction buffers. Most of these only involve inserting
  // calls to Kitsune-specific intrinsic and other relatively simple
  // instructions to dedicated basic blocks.
  Value *numPartials = computeNumPartialReductions(*bbAllocPartials, tapirLoop);
  for (ReductionInfo &reduction : reductions) {
    reduction.numPartials = numPartials;
    reduction.partials = allocPartialsBuffer(*bbAllocPartials, reduction);

    initPartialsBuffer(*bbAllocPartials, loop, reduction);
    reduceIntoPartialsBuffer(*outerLoop, loop, reduction);
    genFinalReduction(*bbReducePartials, reduction);
    freePartialsBuffer(*bbFreePartials, reduction);
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
  BranchInst *br = loop.getLoopGuardBranch();
  assert(br && "Inner loop must have a guard");
  assert(br->getNumSuccessors() == 2 &&
         "Inner loop guard must have two successors");
  assert(br->getSuccessor(0) == *succ_begin(loop.getExitBlock()) &&
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
  return annotateLoopAsPrepared(*outerLoop, /*hasChanged=*/true);
}

template <typename... Args>
static bool complain(const Loop &loop, DiagID diag, Args &&...args) {
  emitDiagnostic(loop, diag, args...);
  return false;
}

// Check that the reduction loop can be transformed. We require it to be in a
// very specific form. Some of the constraints here could, potentially, be
// relaxed in the future. Returns false if any of the preconditions are not
// satisfied.
static bool check(TapirLoopInfo &tapirLoop, DominatorTree &dt, LoopInfo &li,
                  PredicatedScalarEvolution &pse) {
  auto isInLoop = [](const Instruction &inst, const Loop &loop,
                     const LoopInfo &li) -> bool {
    if (Loop *l = li.getLoopFor(inst.getParent()))
      if (loop.contains(l))
        return true;
    return false;
  };

  auto anyPredecessorDoesNotReattach = [](const BasicBlock &latch) -> bool {
    return llvm::any_of(predecessors(&latch), [](const BasicBlock *bb) {
      return isa<ReattachInst>(bb->getTerminator());
    });
  };

  auto isTerminatorCondBr = [](const BasicBlock &bb) -> bool {
    if (auto *br = dyn_cast<BranchInst>(bb.getTerminator()))
      return !br->isUnconditional();
    return false;
  };

  auto getConvergentOpIfAny = [](const Loop &loop) -> Instruction * {
    for (BasicBlock *bb : loop.getBlocks())
      for (Instruction &inst : *bb)
        if (auto *call = dyn_cast<CallBase>(&inst))
          if (call->isConvergent())
            return &inst;
    return nullptr;
  };

  auto getDetachInsts = [](const Loop &loop) -> SmallVector<DetachInst *, 1> {
    SmallVector<DetachInst *, 1> detachInsts;
    for (BasicBlock *bb : getBlocksNotInSubLoops(loop))
      for (Instruction &inst : *bb)
        if (auto *detachInst = dyn_cast<DetachInst>(&inst))
          detachInsts.push_back(detachInst);
    return detachInsts;
  };

  const Loop &loop = *tapirLoop.getLoop();
  Task &task = *tapirLoop.getTask();

  // This is an intentional repeat of the conditional already checked by the
  // assertion above. But it is critical enough that we want this to fail even
  // on non-assert builds. While it is unlikely that this pass will ever be part
  // of a full-blown production compiler that will be built without assertions,
  // we keep it around anyway.
  if (!loop.isLoopSimplifyForm())
    return complain(loop, DiagID::ErrLoopNotSimplifyForm);

  // It is not clear if we strictly require this, but we do tend to run
  // formLCSSA recursively before tapir loop transformations, so it is probably
  // worth requiring.
  if (!loop.isLCSSAForm(dt))
    return complain(loop, DiagID::ErrLoopNotLCSSAForm);

  if (!tapirLoop.hasPrimaryInduction())
    return complain(loop, DiagID::ErrTapirLoopNoPrimaryIV);

  for (auto &[iv, ivDescr] : *tapirLoop.getInductionVars()) {
    for (User *user : iv->users()) {
      if (auto *inst = dyn_cast<Instruction>(user)) {
        if (!isInLoop(*inst, loop, li))
          return complain(loop, DiagID::ErrTapirLoopIVUsedOutsideLoop,
                          getName(*user));
      } else {
        return complain(
            loop, DiagID::ErrGeneric,
            "Use of tapir loop induction variable is not an instruction");
      }
    }
  }

  // TODO:? It is not clear why this is an issue. It was "inherited" from the
  // implementation of the tapir strip-mining pass.
  if (loop.getHeader()->hasAddressTaken())
    return complain(loop, DiagID::ErrGeneric,
                    "tapir reduction loop header has its address taken");

  // Since the loop is guaranteed to be in loop-simplify form, a unique latch
  // is guaranteed to exist.
  BasicBlock *latch = loop.getLoopLatch();
  if (!anyPredecessorDoesNotReattach(*latch))
    return complain(loop, DiagID::ErrGeneric,
                    "tapir reduction loop body does not reattach");

  // TODO?: It is not clear why we need the check on the terminator of the
  // loop latch. We should either remove the check if it is not needed, or
  // ensure that it can never happen - perhaps by running the loop-rotate pass -
  // then remove this comment.
  if (!isTerminatorCondBr(*latch))
    return complain(loop, DiagID::ErrGeneric,
                    "tapir reduction loop latch is not a conditional branch");

  if (!tapirLoop.getTripCount())
    return complain(loop, DiagID::ErrTapirLoopNoFiniteTripCount);

  // Only loops with a computable trip count can be transformed. The trip count
  // must be an integer - it is not clear if we can support non-integer trip
  // counts in the future.
  const SCEV *scevBC = tapirLoop.getBackedgeTakenCount(pse);
  if (isa<SCEVCouldNotCompute>(scevBC) || !scevBC->getType()->isIntegerTy())
    return complain(loop, DiagID::ErrGeneric,
                    "could not compute SCEV for backedge count");

  const SCEV *scevTC = tapirLoop.getExitCount(scevBC, pse);
  if (isa<SCEVCouldNotCompute>(scevTC))
    return complain(loop, DiagID::ErrGeneric,
                    "could not compute SCEV for trip count");

  // Loop-simplify form does not imply a unique exiting block, only a unique
  // latch. We currently do not support conditional exits in parallel reduction
  // loops.
  if (!loop.getExitingBlock())
    return complain(loop, DiagID::ErrGeneric,
                    "tapir reduction loop does not have unique exiting block");

  if (!loop.isSafeToClone())
    return complain(loop, DiagID::ErrTapirLoopNotSafeToClone);

  if (Instruction *inst = getConvergentOpIfAny(loop))
    return complain(loop, DiagID::ErrTapirLoopConvergent, *inst);

  SmallVector<DetachInst *, 1> detachInsts = getDetachInsts(loop);
  if (detachInsts.size() != 1)
    return complain(
        loop, DiagID::ErrGeneric,
        "tapir reduction loop must have a single detach instruction");

  DetachInst *di = detachInsts.front();
  if (di->getDetached() != task.getEntry())
    return complain(loop, DiagID::ErrGeneric,
                    "tapir reduction loop task entry does not match block "
                    "detached from header");

  SmallVector<Instruction *, 1> reattaches;
  SmallVector<BasicBlock *, 4> ehBlocksToClone;
  SmallPtrSet<BasicBlock *, 4> ehBlockPreds;
  SmallPtrSet<LandingPadInst *, 1> inlinedLPads;
  SmallVector<Instruction *, 1> detachedRethrows;

  AnalyzeTaskForSerialization(&task, reattaches, ehBlocksToClone, ehBlockPreds,
                              inlinedLPads, detachedRethrows);

  // We currently do not support exceptions within reduction loops.
  if (di->hasUnwindDest() || !ehBlocksToClone.empty() ||
      !ehBlockPreds.empty() || !inlinedLPads.empty() ||
      !detachedRethrows.empty())
    return complain(loop, DiagID::ErrTapirLoopThrowsException);

  if (reattaches.size() != 1)
    return complain(
        loop, DiagID::ErrGeneric,
        "tapir reduction loop must have a single reattach instruction");

  // If this is not a top-level tapir loop, it is nested within another tapir
  // loop.
  // FIXME: Add support for reductions in nested tapir loops.
  if (!isTopLevelTapirLoop(loop))
    return complain(loop, DiagID::ErrGeneric,
                    "tapir reduction loop must be a top-level loop");

  return true;
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
  // serial tapir target. Calls to the Kitsune reduce intrinsics will be
  // lowered in a separate pass.
  Loop &loop = *tapirLoop.getLoop();
  return annotateLoopAsPrepared(loop, /*hasChanged=*/false);
}

static bool prepare(Loop &loop, DominatorTree &dt, LoopInfo &li,
                    MemorySSA &mssa, OptimizationRemarkEmitter &ore,
                    ScalarEvolution &se, TaskInfo &ti) {
  LLVM_DEBUG(dbgs() << "PrepareReduction: Preparing loop '" << getName(loop)
                    << "'\n");

  // Even if the loop is recognized as a tapir loop, if it does not have the
  // correct structure, the transformation that must be performed by this pass
  // will be difficult, if not impossible to perform. Therefore, check this
  // early, and fail immediately. See the comment above the call to check() for
  // a discussion on why we choose to fail instead of producing working, even if
  // slow, code in such cases.
  Task *task = getTaskIfTapirLoopStructure(&loop, &ti);
  if (!task) {
    complain(loop, DiagID::ErrTapirLoopNoTask);
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
  if (!check(tapirLoop, dt, li, pse))
    exitOnError();

  TTID tt = *getTargetAttr(loop);
  if (tt == TTID::Serial)
    return prepareForSerial(tapirLoop);
  else if (isGPUTT(tt))
    return prepareForGPU(tapirLoop, dt, li, mssa, se, ti);
  else
    return prepareForCPU(tapirLoop, dt, li, mssa, se, ti);
}

PreservedAnalyses PrepareReductionLoopsPass::run(Function &f,
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
    LLVM_DEBUG(dbgs() << "PrepareReduction: Found loop '" << getName(loop)
                      << "'\n");

    bool shouldTransform = isTapirLoop(loop) && hasReductionAttr(loop) &&
                           !hasReductionPreparedAttr(loop);
    if (shouldTransform)
      changed |= prepare(loop, dt, li, mssa, ore, se, ti);
  }

  if (!changed)
    return PreservedAnalyses::all();
  return getLoopPassPreservedAnalyses();
}

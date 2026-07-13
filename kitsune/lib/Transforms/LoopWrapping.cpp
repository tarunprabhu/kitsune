//===- LoopWrapping.cpp - Utilities to wrap loops with other loops --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The utilities here wrap tapir loops with other loops. These are essentially
// helpers for other transformations that are similar to tiling/strip-mining but
// are very specific to tapir loops.
//
// These are also unusual in that they intentionally do not guarantee that the
// resulting loop nest will preserve the behavior of the original loop. It is
// up to the caller to adjust trip counts appropriately.
//
//===----------------------------------------------------------------------===//

#include "LoopWrapping.h"
#include "kitsune/Core/BasicBlockUtils.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/InstUtils.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/TapirLoopUtils.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

#define DEBUG_TYPE "loop-wrap"

using namespace llvm;

namespace {

/// Implementation class that wraps a tapir loop with another tapir loop.
class LoopWrapImpl {
private:
  DominatorTree &dt;
  LoopInfo &li;

  DomTreeUpdater dtu;
  MemorySSAUpdater mssau;

private:
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

public:
  LoopWrapImpl(DominatorTree &dt, LoopInfo &li, MemorySSA &mssa)
      : dt(dt), li(li), dtu(dt, DomTreeUpdater::UpdateStrategy::Eager),
        mssau(&mssa) {}

  Loop *runOn(const TapirLoopInfo &tapirLoop);
};

} // namespace

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
BasicBlock *LoopWrapImpl::genOuterPreheaderBlock(Loop &loop) {
  LLVM_DEBUG(dbgs() << "LoopWrap:   Create preheader block\n");

  BasicBlock *ph = loop.getLoopPreheader();
  BasicBlock *bb = SplitBlock(ph, ph->getTerminator(), &dtu, &li, &mssau,
                              "wrap.ph", /*Before=*/false);

  // At this point, we have created the outer loop preheader *after* the
  // original loop preheader. Split this to get a new preheader for the inner
  // loop.
  (void)SplitBlock(bb, bb->getTerminator(), &dtu, &li, &mssau,
                   "wrap.inner.ph.new", /*Before=*/false);

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
BasicBlock *LoopWrapImpl::genOuterHeaderBlock(Loop &loop, BasicBlock &outerPh) {
  LLVM_DEBUG(dbgs() << "LoopWrap:   Create header block\n");

  BasicBlock *bb = SplitBlock(&outerPh, outerPh.getTerminator(), &dtu, &li,
                              &mssau, "wrap.header", /*Before=*/false);

  // Add a phi node to this header. This will be the primary induction variable
  // of the outer loop.
  Type *ivTy = loop.getCanonicalInductionVariable()->getType();
  PHINode *iv =
      PHINode::Create(ivTy, /*NumReserved=*/2, "wrap.iv", bb->begin());
  Constant *zero = ConstantInt::getSigned(ivTy, 0);
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
BasicBlock *LoopWrapImpl::genOuterReattachBlock(Loop &loop) {
  LLVM_DEBUG(dbgs() << "LoopWrap:   Create reattach block\n");

  // Split the loop exit block. This results in two blocks. The first, which is
  // returned from SplitBlock(), will be an empty block corresponding to
  // LoopExitNew in the figure above. LoopExit will remain essentially
  // unchanged.
  BasicBlock *loopExit = getExitBlockFromLatch(loop);
  (void)SplitBlock(loopExit, loopExit->begin(), &dtu, &li, &mssau,
                   "wrap.inner.exit.new", /*Before=*/true);

  // Split the loop exit block again. SplitBlock() will once again return an
  // empty block while LoopExit will remain effectively unchanged.
  BasicBlock *bb = SplitBlock(loopExit, loopExit->begin(), &dtu, &li, &mssau,
                              "wrap.reattach", /*Before=*/true);

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
BasicBlock *LoopWrapImpl::genOuterLatchBlock(Loop &loop,
                                             BasicBlock &outerReattach) {
  LLVM_DEBUG(dbgs() << "LoopWrap:   Create latch block\n");

  BasicBlock *bb =
      SplitBlock(&outerReattach, outerReattach.getTerminator(), &dtu, &li,
                 &mssau, "wrap.latch", /*Before=*/false);

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
BasicBlock *LoopWrapImpl::genOuterExitBlock(Loop &loop, BasicBlock &outerLatch,
                                            BasicBlock &outerHeader) {
  LLVM_DEBUG(dbgs() << "LoopWrap:   Create exit block\n");

  BasicBlock *bb = SplitBlock(&outerLatch, outerLatch.getTerminator(), &dtu,
                              &li, &mssau, "wrap.exit", /*Before=*/false);

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
BasicBlock *LoopWrapImpl::genInnerLoopGuardBlock(Loop &loop,
                                                 BasicBlock &outerHeader) {
  LLVM_DEBUG(dbgs() << "LoopWrap:   Create loop guard block\n");

  Instruction *term = outerHeader.getTerminator();
  BasicBlock *bb = SplitBlock(&outerHeader, term, &dtu, &li, &mssau,
                              "wrap.inner.guard.new", /*Before=*/false);

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
BasicBlock *LoopWrapImpl::genInnerLoopEndBlock(Loop &loop,
                                               BasicBlock &outerReattach,
                                               BasicBlock &innerGuard) {
  auto sanityCheck = [](Loop &loop) {
    assert(loop.getLoopPreheader() && "Inner loop must have a preheader");
  };

  LLVM_DEBUG(dbgs() << "LoopWrap:   Create loop end block\n");
  sanityCheck(loop);

  LLVMContext &ctx = getContext(loop);
  Instruction *term = outerReattach.getTerminator();
  BasicBlock *innerEnd = SplitBlock(&outerReattach, term, &dtu, &li, &mssau,
                                    "wrap.inner.end.new", /*Before=*/true);

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
void LoopWrapImpl::genOuterLoopInsts(BasicBlock &outerLatch,
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

  LLVM_DEBUG(dbgs() << "LoopWrap:   Add latch instructions\n");
  sanityCheck(outerLatch, outerHeader);

  BranchInst *outerBr = cast<BranchInst>(outerLatch.getTerminator());
  IRBuilder<> builder(outerBr);

  PHINode *outerIV = cast<PHINode>(outerHeader.begin());
  Constant *one = ConstantInt::getSigned(outerIV->getType(), 1);
  Value *outerInc = builder.CreateAdd(outerIV, one, "wrap.iv.inc");

  Instruction *innerInc = getPrimaryIVInc(tapirLoop);
  cast<Instruction>(outerInc)->copyIRFlags(innerInc);

  outerIV->addIncoming(outerInc, &outerLatch);

  Value *tc = tapirLoop.getTripCount();
  Value *outerCmp = builder.CreateICmpEQ(outerInc, tc, "wrap.iv.cmp");
  outerBr->setCondition(outerCmp);
}

// Generate loop metadata for the outer loop. This will be a clone of the
// metadata of the loop, \p loop being transformed.
void LoopWrapImpl::genOuterLoopMD(BasicBlock &outerLatch, Loop &loop) {
  LLVM_DEBUG(dbgs() << "LoopWrap:   Create loop metadata\n");

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
Loop *LoopWrapImpl::genOuterLoopObject(
    Loop &loop, BasicBlock &outerPh, BasicBlock &outerHeader,
    BasicBlock &outerReattach, BasicBlock &outerLatch, BasicBlock &outerExit,
    BasicBlock &innerGuard, BasicBlock &innerEnd) {
  auto sanityCheck = [](Loop &loop, BasicBlock &outerHeader,
                        BasicBlock &outerReattach, BasicBlock &outerLatch,
                        BasicBlock &outerExit, BasicBlock &innerGuard,
                        BasicBlock &innerEnd) {
    BasicBlock *ph = loop.getLoopPreheader();
    BasicBlock *exit = getExitBlockFromLatch(loop);

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
           "First successor of inner loop guard must be loop end block");
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

  auto addBlockToLoop = [](Loop &loop, BasicBlock &bb, LoopInfo &li) {
    if (loop.getParentLoop()) {
      li.changeLoopFor(&bb, &loop);
      loop.addBlockEntry(&bb);
    } else {
      loop.addBasicBlockToLoop(&bb, li);
    }
  };

  LLVM_DEBUG(dbgs() << "LoopWrap:   Create loop object\n");
  sanityCheck(loop, outerHeader, outerReattach, outerLatch, outerExit,
              innerGuard, innerEnd);

  Loop *outerLoop = li.AllocateLoop();
  if (Loop *parentLoop = loop.getParentLoop())
    parentLoop->replaceChildLoopWith(&loop, outerLoop);
  else
    li.changeTopLevelLoop(&loop, outerLoop);
  outerLoop->addChildLoop(&loop);

  // Add blocks to the outer loop. We add them in roughly the same order as the
  // figure above just to keep things somewhat organized. The blocks in the
  // inner loop must also be added to the new outer loop. These don't have to be
  // added to the parents of the outer loop (if any) since they should already
  // be present there, so we just use addBlockEntry().
  addBlockToLoop(*outerLoop, outerHeader, li);
  addBlockToLoop(*outerLoop, innerGuard, li);
  addBlockToLoop(*outerLoop, *loop.getLoopPreheader(), li);
  for (BasicBlock *bb : loop.getBlocks())
    outerLoop->addBlockEntry(bb);
  addBlockToLoop(*outerLoop, *getExitBlockFromLatch(loop), li);
  addBlockToLoop(*outerLoop, innerEnd, li);
  addBlockToLoop(*outerLoop, outerReattach, li);
  addBlockToLoop(*outerLoop, outerLatch, li);

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
Loop *LoopWrapImpl::runOn(const TapirLoopInfo &tapirLoop) {
  LLVM_DEBUG(dbgs() << "LoopWrap: Generate outer loop\n");

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
  // Yes, this is obviously "wrong" since the behavior of the resulting loop
  // nest will be different from the original. But we have left it up to the
  // caller to fix the trip counts appropriately.
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

  LLVM_DEBUG(dbgs() << "LoopWrap: Done generating outer loop\n");
  LLVM_DEBUG(dbgs() << "LoopWrap:\n" << *outerLoop);

  return outerLoop;
}

template <typename... Args>
static bool complain(const Loop &loop, DiagID diag, Args &&...args) {
  emitDiagnostic(loop, diag, args...);
  return false;
}

bool llvm::checkTapirLoopSafeToWrap(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                                    LoopInfo &li) {
  auto anyPredecessorDoesNotReattach = [](const BasicBlock &latch) -> bool {
    return llvm::any_of(predecessors(&latch), [](const BasicBlock *bb) {
      return isa<ReattachInst>(bb->getTerminator());
    });
  };

  auto getConvergentOpIfAny = [](const Loop &loop) -> Instruction * {
    for (BasicBlock *bb : loop.getBlocks())
      for (Instruction &inst : *bb)
        if (auto *call = dyn_cast<CallBase>(&inst))
          if (call->isConvergent())
            return &inst;
    return nullptr;
  };

  const Loop &loop = *tapirLoop.getLoop();
  Task &task = *tapirLoop.getTask();

  if (!loop.isLoopSimplifyForm())
    return complain(loop, DiagID::ErrLoopNotSimplifyForm);

  // It is not clear if we strictly require this, but we do tend to run the
  // LCSSA pass before tapir loop transformation passes, so we check for it.
  if (!loop.isLCSSAForm(dt))
    return complain(loop, DiagID::ErrLoopNotLCSSAForm);

  // We don't allow early termination in parallel loops. One would, therefore,
  // expect that tapir loops would have a unique exit block. However, some
  // transformation passes may result in the code having non-unique exit blocks.
  // This is only ok if those exit blocks are dead-ends i.e. blocks where
  // all paths out of the block will lead to an unreachable instruction.
  if (!getUniqueNonDeadEndExitBlock(loop))
    return complain(loop, DiagID::ErrTapirLoopNoUniqueNonDeadEndExitBlock);

  if (getNumIndVars(loop) > 1)
    return complain(loop, DiagID::ErrTapirLoopIVMultiple);

  if (!loop.getCanonicalInductionVariable())
    return complain(loop, DiagID::ErrTapirLoopIVNotCanonical);

  for (auto &[iv, ivDescr] : *tapirLoop.getInductionVars()) {
    for (User *user : iv->users())
      if (!isa<Instruction>(user))
        return complain(loop, DiagID::ErrTapirLoopIVUseNotInst);

    if (isUsedOutsideLoop(*iv, loop, li))
      return complain(loop, DiagID::ErrTapirLoopIVUsedOutsideLoop);
  }

  // TODO:? It is not clear why this is an issue. It was "inherited" from the
  // implementation of the tapir strip-mining pass.
  if (loop.getHeader()->hasAddressTaken())
    return complain(loop, DiagID::ErrTapirLoopHeaderAddressTaken);

  // Since the loop is guaranteed to be in loop-simplify form, a unique latch
  // is guaranteed to exist.
  BasicBlock *latch = loop.getLoopLatch();
  if (!anyPredecessorDoesNotReattach(*latch))
    return complain(loop, DiagID::ErrTapirLoopBodyDoesNotReattach);

  // Most transformation passes except that the terminator of the tapir loop
  // latch is a conditional branch.
  if (!isCondBr(*latch->getTerminator()))
    return complain(loop, DiagID::ErrTapirLoopBlockTerminator, "latch",
                    "conditional branch");

  // We check this late because one reason for the failure to compute a finite
  // trip count is that the terminator of the loop latch is not a conditional
  // branch. By allowing that error to be emitted, we have a better idea of why
  // this check failed.
  if (!tapirLoop.hasTripCount())
    return complain(loop, DiagID::ErrTapirLoopNoFiniteTripCount);

  if (!loop.isSafeToClone())
    return complain(loop, DiagID::ErrTapirLoopNotSafeToClone);

  if (Instruction *inst = getConvergentOpIfAny(loop))
    return complain(loop, DiagID::ErrTapirLoopConvergent, *inst);

  const DetachInst *di = getUniqueInstInLoopOnly<DetachInst>(loop);
  if (!di)
    return complain(loop, DiagID::ErrTapirLoopNoUniqueDetachInst);
  else if (di->getDetached() != task.getEntry())
    return complain(loop, DiagID::ErrTapirLoopTaskEntryMismatch);

  SmallVector<Instruction *, 1> reattaches;
  SmallVector<BasicBlock *, 4> ehBlocksToClone;
  SmallPtrSet<BasicBlock *, 4> ehBlockPreds;
  SmallPtrSet<LandingPadInst *, 1> inlinedLPads;
  SmallVector<Instruction *, 1> detachedRethrows;

  AnalyzeTaskForSerialization(&task, reattaches, ehBlocksToClone, ehBlockPreds,
                              inlinedLPads, detachedRethrows);

  // We currently do not support exceptions within tapir loops.
  if (di->hasUnwindDest() || !ehBlocksToClone.empty() ||
      !ehBlockPreds.empty() || !inlinedLPads.empty() ||
      !detachedRethrows.empty())
    return complain(loop, DiagID::ErrTapirLoopThrowsException);

  if (reattaches.size() != 1)
    return complain(loop, DiagID::ErrTapirLoopNoUniqueReattachInst);

  return true;
}

Loop *llvm::wrapWithTapirLoop(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                              LoopInfo &li, MemorySSA &mssa) {
  return LoopWrapImpl(dt, li, mssa).runOn(tapirLoop);
}

BranchInst *llvm::getWrappedLoopGuardBranch(Loop &loop) {
  BasicBlock *ph = loop.getLoopPreheader();
  assert(ph && "Wrapped loop must have a preheader");

  BasicBlock *guard = ph->getUniquePredecessor();
  assert(guard && "Wrapped loop preheader must have a unique predecessor");

  return dyn_cast<BranchInst>(guard->getTerminator());
}

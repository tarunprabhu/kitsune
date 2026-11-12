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
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/TapirLoopUtils.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

#define DEBUG_TYPE "loop-wrap"

using namespace llvm;

namespace {

// The blocks in a loop. This is simply for convenience.
class LoopBlocks {
public:
  BasicBlock *guard = nullptr;
  BasicBlock *ph = nullptr;
  BasicBlock *header = nullptr;
  BasicBlock *reattach = nullptr;
  BasicBlock *latch = nullptr;
  BasicBlock *exit = nullptr;
  BasicBlock *bypass = nullptr;

public:
  LoopBlocks() = default;
  LoopBlocks(const LoopBlocks &) = delete;
  LoopBlocks(LoopBlocks &&) = delete;

  LoopBlocks(const Loop &loop) {
    guard = getLoopGuardBlock(loop, /*allowDeadEndExits=*/true);
    ph = loop.getLoopPreheader();
    header = loop.getHeader();
    reattach = getTapirLoopReattachBlock(loop);
    latch = loop.getLoopLatch();
    exit = getUniqueNonDeadEndExitBlock(loop);
    bypass = getLoopBypassBlock(loop, /*allowDeadEndExits=*/true);
  }
};

// Implementation class that wraps a tapir loop with a serial loop.
//
// The CFG of \p tapirLoop is expected to be as shown below.
//
//     LoopGuard
//     LoopPreheader
//         LoopHeader
//         <LoopBlocks>
//         LoopLatch
//     LoopExit
//     LoopBypass
//
// The CFG after the loop is wrapped will be transformed as shown below.
//
//     LoopGuard1
//     OuterGuard
//     OuterPreheader
//         OuterHeader
//         LoopGuard2
//         LoopPreheader
//             LoopHeader
//             <LoopBlocks>
//             LoopLatch
//         LoopExit
//         LoopBypass
//         OuterReattach
//         OuterLatch
//     OuterExit
//     OuterBypass
//
// For more information, see the API documentation for wrapWithSerialLoop.
class WrapLoopBase {
protected:
  DominatorTree &dt;
  LoopInfo &li;
  ScalarEvolution &se;
  TaskInfo &ti;

  DomTreeUpdater dtu;
  MemorySSAUpdater mssau;

protected:
  LoopBlocks populateLoopBlocks(Loop &loop);

  void genGuardBlock(Loop &loop, LoopBlocks &bbs, LoopBlocks &bbsGen);
  void genPreheaderBlock(Loop &loop, const LoopBlocks &bbs, LoopBlocks &bbsGen);
  void genHeaderBlock(Loop &loop, const LoopBlocks &bbs, LoopBlocks &bbsGen);
  void genReattachBlock(Loop &loop, const LoopBlocks &bbs, LoopBlocks &bbsGen);
  void genLatchBlock(Loop &loop, const LoopBlocks &bbs, LoopBlocks &bbsGen);
  void genExitBlock(Loop &loop, const LoopBlocks &bbs, LoopBlocks &bbsGen);
  void genBypassBlock(Loop &loop, const LoopBlocks &bbs, LoopBlocks &bbsGen);

  virtual void genGuardInsts(const Loop &loop, const LoopBlocks &bbs,
                             const LoopBlocks &bbsGen);
  virtual void genHeaderInsts(const Loop &loop, const LoopBlocks &bbs,
                              const LoopBlocks &bbsGen);
  virtual void genLatchInsts(const Loop &loop, const LoopBlocks &bbs,
                             const LoopBlocks &bbsGen);

  virtual void genReattachInsts(const Loop &loop, const LoopBlocks &bbs,
                                const LoopBlocks &bbsGen) = 0;
  virtual void genBypassInsts(const Loop &loop, const LoopBlocks &bbs,
                              const LoopBlocks &bbsGen) = 0;
  virtual void genWrapperLoopMD(const Loop &loop, const LoopBlocks &bbsGen) = 0;

  Loop *genWrapperLoopObject(Loop &loop, const LoopBlocks &bbs,
                             const LoopBlocks &bbsGen);

protected:
  WrapLoopBase(DominatorTree &dt, LoopInfo &li, MemorySSA &mssa,
               ScalarEvolution &se, TaskInfo &ti)
      : dt(dt), li(li), se(se), ti(ti),
        dtu(dt, DomTreeUpdater::UpdateStrategy::Eager), mssau(&mssa) {}

public:
  Loop *runOn(const TapirLoopInfo &tapirLoop);
};

// Implementation class to wrap a tapir loop with a serial loop
class WrapWithSerialLoop : public WrapLoopBase {
protected:
  virtual void genReattachInsts(const Loop &loop, const LoopBlocks &bbs,
                                const LoopBlocks &bbsGen) override final;
  virtual void genBypassInsts(const Loop &loop, const LoopBlocks &bbs,
                              const LoopBlocks &bbsGen) override final;
  virtual void genWrapperLoopMD(const Loop &loop,
                                const LoopBlocks &bbsGen) override final;

public:
  WrapWithSerialLoop(DominatorTree &dt, LoopInfo &li, MemorySSA &mssa,
                     ScalarEvolution &se, TaskInfo &ti)
      : WrapLoopBase(dt, li, mssa, se, ti) {}
};

// Implementation class to wrap a tapir loop with another tapir loop
class WrapWithTapirLoop : public WrapLoopBase {
private:
  Value *getWrapperSyncRegion(BasicBlock &guard);

protected:
  virtual void genGuardInsts(const Loop &loop, const LoopBlocks &bbs,
                             const LoopBlocks &bbsGen) override final;
  virtual void genHeaderInsts(const Loop &loop, const LoopBlocks &bbs,
                              const LoopBlocks &bbsGen) override final;
  virtual void genReattachInsts(const Loop &loop, const LoopBlocks &bbs,
                                const LoopBlocks &bbsGen) override final;
  virtual void genBypassInsts(const Loop &loop, const LoopBlocks &bbs,
                              const LoopBlocks &bbsGen) override final;

  virtual void genWrapperLoopMD(const Loop &loop,
                                const LoopBlocks &bbsGen) override final;

public:
  WrapWithTapirLoop(DominatorTree &dt, LoopInfo &li, MemorySSA &mssa,
                    ScalarEvolution &se, TaskInfo &ti)
      : WrapLoopBase(dt, li, mssa, se, ti) {}
};

} // namespace

void WrapLoopBase::genGuardBlock(Loop &loop, LoopBlocks &bbs,
                                 LoopBlocks &bbsGen) {
  // Get the point at which to split the original loop guard block to construct
  // the LoopGuard1 and LoopGuard2 blocks in the CFG diagram above. In general,
  // this will try to keep the original terminator and the instructions that
  // comprise the guard condition in LoopGuard2, and everything else in
  // LoopGuard1. At the very least, this will return the terminator of the guard
  // block \p bb.
  auto getGuardSplitPoint = [](BasicBlock &guard) -> Instruction * {
    BranchInst *br = cast<BranchInst>(guard.getTerminator());
    Value *cond = br->getCondition();
    if (!isa<Instruction>(cond))
      return br;

    // We generally want all the instructions in LoopGuard2 to be directly
    // related to the condition of the branch. If the condition is an
    // instruction, but not the immediate predecessor of the branch, just split
    // at the branch. The whole point of finding a different split point is to
    // keep the IR organized in a way that is marginally easier to read if
    // needed.
    CmpInst *cmp = dyn_cast<CmpInst>(cond);
    if (!cmp || br->getPrevNode() != cmp)
      return br;

    return cmp;
  };

  LLVM_DEBUG(dbgs() << "WrapLoop:   Create guard block\n");

  assert(bbs.guard && "Loop has guard block");
  assert(isa<BranchInst>(bbs.guard->getTerminator()) &&
         "Terminator of guard block is a branch instruction");
  assert(cast<BranchInst>(bbs.guard->getTerminator()) &&
         "Terminator of guard block is a conditional branch instruction");

  BasicBlock *origGuard = bbs.guard;

  // We try to group a "reasonable" set of instructions into what will become
  // the guard of the loop being wrapped.
  Instruction *splitPt = getGuardSplitPoint(*origGuard);

  // Split the original loop guard block. Since we have passed `Before=false` to
  // SplitBlock, the instructions before the split point will remain in the
  // block, while the split point and any instructions after it will be moved
  // to a new block. This new block is returned by SplitBlock. The new block
  // will be the guard block of the original loop. The remaining instructions
  // will be in the predecessor of the wrapper loop guard that will eventually
  // be added, i.e. they will be present in LoopGuard1 in the CFG diagram in the
  // documentation for this class. Since the guard of the original loop has
  // effectively now changed, we must update the recorded guard.
  bbs.guard = SplitBlock(origGuard, splitPt, &dtu, &li, &mssau, "guard");

  // Splitting the guard block at the beginning will generate an empty block
  // between LoopGuard1 that was created above and, what is now LoopGuard2. This
  // will be the guard block for the wrapper loop.
  BasicBlock *guard = bbs.guard;
  bbsGen.guard =
      splitBlockBefore(guard, guard->begin(), &dtu, &li, &mssau, "wrap.guard");
}

void WrapLoopBase::genPreheaderBlock(Loop &loop, const LoopBlocks &bbs,
                                     LoopBlocks &bbsGen) {
  LLVM_DEBUG(dbgs() << "WrapLoop:   Create preheader block\n");

  assert(bbsGen.guard && "Generated wrapped loop guard block");

  BasicBlock *guard = bbsGen.guard;
  Instruction *term = guard->getTerminator();
  bbsGen.ph = SplitBlock(guard, term, &dtu, &li, &mssau, "wrap.ph");
}

void WrapLoopBase::genHeaderBlock(Loop &loop, const LoopBlocks &bbs,
                                  LoopBlocks &bbsGen) {
  LLVM_DEBUG(dbgs() << "WrapLoop:   Create header block\n");

  assert(bbsGen.ph && "Generated wrapped loop preheader block");

  BasicBlock *ph = bbsGen.ph;
  Instruction *term = ph->getTerminator();
  bbsGen.header = SplitBlock(ph, term, &dtu, &li, &mssau, "wrap.header");
}

void WrapLoopBase::genReattachBlock(Loop &loop, const LoopBlocks &bbs,
                                    LoopBlocks &bbsGen) {
  LLVM_DEBUG(dbgs() << "WrapLoop:   Create reattach block\n");

  assert(bbs.bypass && "Loop has bypass block");
  assert(isa<SyncInst>(bbs.bypass->getTerminator()) &&
         "Terminator of loop bypass block is a sync instruction");

  BasicBlock *bypass = bbs.bypass;
  SyncInst *sync = cast<SyncInst>(bypass->getTerminator());
  BasicBlock *cont = sync->getSuccessor(0);

  bbsGen.reattach =
      splitBlockBefore(cont, cont->begin(), &dtu, &li, &mssau, "wrap.reattach");
}

void WrapLoopBase::genLatchBlock(Loop &loop, const LoopBlocks &bbs,
                                 LoopBlocks &bbsGen) {
  LLVM_DEBUG(dbgs() << "WrapLoop:   Create latch block\n");

  assert(bbsGen.reattach && "Generated wrapped loop reattach block");

  BasicBlock *reattach = bbsGen.reattach;
  Instruction *term = reattach->getTerminator();
  bbsGen.latch = SplitBlock(reattach, term, &dtu, &li, &mssau, "wrap.latch");
}

void WrapLoopBase::genExitBlock(Loop &loop, const LoopBlocks &bbs,
                                LoopBlocks &bbsGen) {
  LLVM_DEBUG(dbgs() << "WrapLoop:   Create exit block\n");

  assert(bbsGen.latch && "Generated wrapped loop latch block");

  BasicBlock *latch = bbsGen.latch;
  Instruction *term = latch->getTerminator();
  bbsGen.exit = SplitBlock(latch, term, &dtu, &li, &mssau, "wrap.loopexit");
}

void WrapLoopBase::genBypassBlock(Loop &loop, const LoopBlocks &bbs,
                                  LoopBlocks &bbsGen) {
  LLVM_DEBUG(dbgs() << "WrapLoop:   Create bypass block\n");

  assert(bbsGen.exit && "Generated wrapped loop exit block");

  BasicBlock *exit = bbsGen.exit;
  Instruction *term = exit->getTerminator();
  bbsGen.bypass = SplitBlock(exit, term, &dtu, &li, &mssau, "wrap.bypass");
}

void WrapLoopBase::genGuardInsts(const Loop &loop, const LoopBlocks &bbs,
                                 const LoopBlocks &bbsGen) {
  LLVM_DEBUG(dbgs() << "WrapLoop:   Populate guard block\n");

  assert(bbsGen.guard && "Generated wrapper loop guard block");
  assert(bbsGen.ph && "Generated wrapper loop preheader block");
  assert(bbsGen.bypass && "Generated wrappre loop bypass block");
  assert(succ_size(bbsGen.guard) == 1 &&
         "Wrapper guard branches unconditionally");
  assert(*succ_begin(bbsGen.guard) == bbsGen.ph &&
         "Wrapper guard branches to the wrapper preheader");

  BasicBlock *guard = bbsGen.guard;
  BasicBlock *ph = bbsGen.ph;
  BasicBlock *bypass = bbsGen.bypass;

  LLVMContext &ctx = guard->getContext();
  Type *i1 = Type::getInt1Ty(ctx);

  Instruction *term = guard->getTerminator();
  Value *cond = ConstantInt::getTrue(i1);
  BranchInst *condBr = BranchInst::Create(ph, bypass, cond);

  ReplaceInstWithInst(term, condBr);

  dt.insertEdge(guard, bypass);
#ifndef NDEBUG
  dt.verify();
#endif // NDEBUG
}

void WrapLoopBase::genHeaderInsts(const Loop &loop, const LoopBlocks &bbs,
                                  const LoopBlocks &bbsGen) {
  LLVM_DEBUG(dbgs() << "WrapLoop:   Populate header block\n");

  assert(bbsGen.ph && "Generated wrapper loop preheader block");
  assert(bbsGen.header && "Generated wrapper loop header block");
  assert(succ_size(bbsGen.ph) == 1 &&
         "Wrapper preheader branches unconditionally");
  assert(*succ_begin(bbsGen.ph) == bbsGen.header &&
         "Wrapper preheader branches to the wrapper header");

  LLVMContext &ctx = getContext(loop);
  Type *i64 = Type::getInt64Ty(ctx);

  BasicBlock *ph = bbsGen.ph;
  BasicBlock *header = bbsGen.header;
  Instruction *term = header->getTerminator();

  PHINode *iv = PHINode::Create(i64, /*NumReservedValues=*/2, "wrap.iv",
                                term->getIterator());
  Value *zero = ConstantInt::get(i64, 0, /*isSigned=*/false);
  iv->addIncoming(zero, ph);
}

void WrapLoopBase::genLatchInsts(const Loop &loop, const LoopBlocks &bbs,
                                 const LoopBlocks &bbsGen) {
  LLVM_DEBUG(dbgs() << "WrapLoop:   Populate latch block\n");

  assert(bbsGen.header && "Generated wrapper loop header block");
  assert(bbsGen.latch && "Generated wrapper loop latch block");
  assert(bbsGen.exit && "Generated wrapper loop exit block");
  assert(succ_size(bbsGen.latch) == 1 &&
         "Wrapper latch branches unconditionally");
  assert(*succ_begin(bbsGen.latch) == bbsGen.exit &&
         "Wrapper latches branches to the exit block");
  assert(bbsGen.header->size() == 2 &&
         "Wrapper header has exactly two instructions");

  assert(getUniquePHI(*bbsGen.header) &&
         "Wrapper header has a single PHI node");
  assert(getUniquePHI(*bbsGen.header)->getType()->isIntegerTy(64) &&
         "Wrapper induction variable is of type i64");

  BasicBlock *header = bbsGen.header;
  BasicBlock *latch = bbsGen.latch;
  BasicBlock *exit = bbsGen.exit;

  PHINode *iv = getUniquePHI(*header);
  Type *ivTy = iv->getType();
  Constant *zero = ConstantInt::get(ivTy, 0, /*isSigned=*/false);
  Constant *one = ConstantInt::get(ivTy, 1, /*isSigned=*/false);

  IRBuilder<> builder(latch->getTerminator());
  Value *inc = builder.CreateNUWAdd(iv, one, "wrap.iv.inc");
  Value *cmp = builder.CreateICmpEQ(inc, zero, "wrap.iv.cmp");
  BranchInst *br = BranchInst::Create(exit, header, cmp);

  iv->addIncoming(inc, latch);
  ReplaceInstWithInst(latch->getTerminator(), br);

  dt.insertEdge(latch, header);
#ifndef NDEBUG
  dt.verify();
#endif // NDEBUG
}

// Generate a loop object for the outer loop with the given blocks. This will
// update the analysis objects in this class, so they should be safe to use
// after this returns.
Loop *WrapLoopBase::genWrapperLoopObject(Loop &loop, const LoopBlocks &bbs,
                                         const LoopBlocks &bbsGen) {
  auto sanityCheck = [](const Loop &loop, const LoopBlocks &bbs,
                        const LoopBlocks &bbsGen) {
    assert(bbs.guard && "Wrapped loop has guard block");
    assert(bbs.ph && "Wrapped loop has preheader block");
    assert(bbs.exit && "Wrapped loop has exit block");
    assert(bbs.bypass && "Wrapped loop has bypass block");

    assert(bbsGen.guard && "Generated wrapper loop guard block");
    assert(bbsGen.ph && "Generated wrapper loop preheader block");
    assert(bbsGen.header && "Generated wrapper loop header block");
    assert(bbsGen.reattach && "Generated wrapper loop reattach block");
    assert(bbsGen.latch && "Generated wrapper loop latch block");
    assert(bbsGen.exit && "Generated wrapper loop exit block");
    assert(bbsGen.bypass && "Generated wrapper loop bypass block");
  };

  auto addBlockToLoop = [](Loop &loop, BasicBlock &bb, LoopInfo &li) {
    if (loop.getParentLoop()) {
      li.changeLoopFor(&bb, &loop);
      loop.addBlockEntry(&bb);
    } else {
      loop.addBasicBlockToLoop(&bb, li);
    }
  };

  LLVM_DEBUG(dbgs() << "WrapLoop:   Create loop object\n");
  sanityCheck(loop, bbs, bbsGen);

#ifndef NDEBUG
  Loop *parentLoop = loop.getParentLoop();
#endif // NDEBUG

  Loop *wrapperLoop = li.AllocateLoop();
  if (Loop *parentLoop = loop.getParentLoop())
    parentLoop->replaceChildLoopWith(&loop, wrapperLoop);
  else
    li.changeTopLevelLoop(&loop, wrapperLoop);
  wrapperLoop->addChildLoop(&loop);

  // Add blocks to the outer loop. We add them in roughly the same order as the
  // figure above just to keep things somewhat organized. The blocks in the
  // inner loop must also be added to the new outer loop. These don't have to be
  // added to the parents of the outer loop (if any) since they should already
  // be present there, so we just use addBlockEntry().
  addBlockToLoop(*wrapperLoop, *bbsGen.header, li);
  addBlockToLoop(*wrapperLoop, *bbs.guard, li);
  addBlockToLoop(*wrapperLoop, *bbs.ph, li);
  for (BasicBlock *bb : loop.getBlocks())
    wrapperLoop->addBlockEntry(bb);
  addBlockToLoop(*wrapperLoop, *bbs.exit, li);
  addBlockToLoop(*wrapperLoop, *bbs.bypass, li);
  addBlockToLoop(*wrapperLoop, *bbsGen.reattach, li);
  addBlockToLoop(*wrapperLoop, *bbsGen.latch, li);

  // Optionally run some sanity checks before updating task analysis since it
  // might help catch issues earlier.
#ifndef NDEBUG
  assert(loop.isLoopSimplifyForm() &&
         "Inner loop must be maintained in loop-simplify form");
  assert(wrapperLoop->isLoopSimplifyForm() &&
         "Outer loop must be generated in loop-simplify form");

  dt.verify();
  li.verify(dt);
  loop.verifyLoop();
  wrapperLoop->verifyLoop();
  if (parentLoop) {
    parentLoop->verifyLoop();
    assert(wrapperLoop->getParentLoop() == parentLoop &&
           "Original parent of loop nest is parent of prepared outer loop");
  }

  const std::vector<Loop *> &subLoops = wrapperLoop->getSubLoops();
  assert(subLoops.size() == 1 && "Wrapped loop has a single subloop");
  assert(subLoops[0] == &loop && "Wrapper loop subloop is the original loop");
#endif // NDEBUG

  // FIXME: Recomputing task analysis for the entire function is wasteful,
  // but there is currently no more efficient way to do this.
  //
  // TaskInfo will have to be recalculated even when wrapping a serial loop
  // around a tapir loop nest.
  Function *f = getFunction(loop);
  ti.recalculate(*f, dt);

#ifndef NDEBUG
  ti.verify(dt);
#endif // NDEBUG

  return wrapperLoop;
}

Loop *WrapLoopBase::runOn(const TapirLoopInfo &tapirLoop) {
  LLVM_DEBUG(dbgs() << "WrapLoop: Generate outer loop\n");

  Loop &loop = *tapirLoop.getLoop();

  // The blocks in the original loop being wrapped.
  LoopBlocks bbs(loop);

  // We first create the core blocks required by the outer loop. At no stage
  // during the creation of these can the CFG be assumed to be "correct" i.e.
  // blocks may be orphaned, induction variables may not have the correct set of
  // incoming values and blocks, conditional branch instructions may have
  // invalid conditions etc. The CFG will be kept "somewhat sane", but that's
  // about it.
  LoopBlocks bbsGen;
  genGuardBlock(loop, bbs, bbsGen);
  genPreheaderBlock(loop, bbs, bbsGen);
  genHeaderBlock(loop, bbs, bbsGen);
  genReattachBlock(loop, bbs, bbsGen);
  genLatchBlock(loop, bbs, bbsGen);
  genExitBlock(loop, bbs, bbsGen);
  genBypassBlock(loop, bbs, bbsGen);

  // We have now created the "core" basic blocks for the outer loop. However,
  // all terminators of the newly created blocks are unconditional branches.
  // There are no backedges from the latch, no phi nodes in the header, and no
  // conditional branch in the guard block.
  //
  // These are fixed, though with yet more placeholders. For instance,
  //
  //   - The bounds of the wrapped loop are set to be [0, 0) with a step of 1.
  //
  //   - The condition of the wrapper guard is set to a boolean so the wrapped
  //     loop is never bypassed.
  //
  // These are expected to be fixed by the caller.
  genGuardInsts(loop, bbs, bbsGen);
  genHeaderInsts(loop, bbs, bbsGen);
  genReattachInsts(loop, bbs, bbsGen);
  genLatchInsts(loop, bbs, bbsGen);
  genBypassInsts(loop, bbs, bbsGen);

  // Generate appropriate loop metadata for the outer loop. This will just be a
  // clone of the metadata of the original loop.
  genWrapperLoopMD(loop, bbsGen);

  // Generate a loop object for the outer loop with the given blocks. This will
  // also update the LoopInfo object, so it should be safe to use it after
  // this.
  Loop *wrapperLoop = genWrapperLoopObject(loop, bbs, bbsGen);

  LLVM_DEBUG(dbgs() << "WrapLoop: Done generating outer loop\n");
  LLVM_DEBUG(dbgs() << "WrapLoop:\n" << *wrapperLoop);

  return wrapperLoop;
}

//==---------------------------- WrapLoopSerial ----------------------------==//

void WrapWithSerialLoop::genReattachInsts(const Loop &loop,
                                          const LoopBlocks &bbs,
                                          const LoopBlocks &bbsGen) {
  // Nothing to be done when wrapping with a serial loop
}

void WrapWithSerialLoop::genBypassInsts(const Loop &loop, const LoopBlocks &bbs,
                                        const LoopBlocks &bbsGen) {
  // Nothing to be done when wrapping with a serial loop
}

// Generate loop metadata for the outer loop. This will clone the metadata of
// \p loop, but any tapir-specific loop attributes will be removed.
void WrapWithSerialLoop::genWrapperLoopMD(const Loop &loop,
                                          const LoopBlocks &bbsGen) {
  LLVM_DEBUG(dbgs() << "WrapLoop:   Create loop metadata\n");

  assert(loop.getLoopID() && "Loop has ID metadata");
  assert(bbsGen.latch && "Generated wrapper loop latch");

  LLVMContext &ctx = getContext(loop);
  MDNode *newMD = makePostTransformationMetadata(ctx, cloneLoopMetadata(loop),
                                                 {tapirLoopAttrNamePrefix}, {});

  BasicBlock *latch = bbsGen.latch;
  Instruction *term = latch->getTerminator();
  term->setMetadata(LLVMContext::MD_loop, newMD);
}

//==--------------------------- WrapLoopParallel ---------------------------==//

Value *WrapWithTapirLoop::getWrapperSyncRegion(BasicBlock &bb) {
  for (Instruction &inst : bb)
    if (auto *call = dyn_cast<CallInst>(&inst))
      if (call->getIntrinsicID() == Intrinsic::syncregion_start)
        return call;

  assert(false && "Wrapped loop guard block defines a sync region");
  return nullptr;
}

void WrapWithTapirLoop::genGuardInsts(const Loop &loop, const LoopBlocks &bbs,
                                      const LoopBlocks &bbsGen) {
  // When wrapping with a parallel loop, most of the basic implementation can
  // still be used, but we need to create a syncregion.
  WrapLoopBase::genGuardInsts(loop, bbs, bbsGen);

  BasicBlock *guard = bbsGen.guard;

  IRBuilder<> builder(&guard->front());
  (void)builder.CreateIntrinsic(Intrinsic::syncregion_start, {},
                                /*FMFSource=*/{}, "wrap.syncreg");
}

void WrapWithTapirLoop::genHeaderInsts(const Loop &loop, const LoopBlocks &bbs,
                                       const LoopBlocks &bbsGen) {
  // In addition to the standard instructions, we need to replace the header
  // terminator with a detach instruction.
  WrapLoopBase::genHeaderInsts(loop, bbs, bbsGen);

  assert(bbsGen.guard && "Generated wrapper latch guard");
  assert(bbsGen.latch && "Generated wrapper latch block");
  assert(succ_size(bbsGen.header) == 1 &&
         "Wrapper header has a single successor");
  assert(*succ_begin(bbsGen.header) == bbs.guard &&
         "Successor of wrapper loop header is wrapped loop guard");

  BasicBlock *header = bbsGen.header;
  BasicBlock *guard = bbsGen.guard;
  BasicBlock *latch = bbsGen.latch;
  BasicBlock *body = bbs.guard;

  Value *syncreg = getWrapperSyncRegion(*guard);
  DetachInst *detach = DetachInst::Create(body, latch, syncreg);

  ReplaceInstWithInst(header->getTerminator(), detach);

  dt.insertEdge(header, latch);
#ifndef NDEBUG
  dt.verify();
#endif // NDEBUG
}

void WrapWithTapirLoop::genReattachInsts(const Loop &loop,
                                         const LoopBlocks &bbs,
                                         const LoopBlocks &bbsGen) {
  LLVM_DEBUG(dbgs() << "WrapLoop:   Populate reattach block\n");

  assert(bbsGen.guard && "Generated guard block");
  assert(bbsGen.reattach && "Generated reattach block");
  assert(bbsGen.latch && "Generated latch block");
  assert(succ_size(bbsGen.reattach) == 1 &&
         "Generated reattach block has a single successor");
  assert(*succ_begin(bbsGen.reattach) == bbsGen.latch &&
         "Successor of reattach is latch");

  BasicBlock *guard = bbsGen.guard;
  BasicBlock *reattach = bbsGen.reattach;
  BasicBlock *latch = bbsGen.latch;

  Value *syncreg = getWrapperSyncRegion(*guard);
  ReattachInst *reattachInst = ReattachInst::Create(latch, syncreg);

  ReplaceInstWithInst(reattach->getTerminator(), reattachInst);
}

void WrapWithTapirLoop::genBypassInsts(const Loop &loop, const LoopBlocks &bbs,
                                       const LoopBlocks &bbsGen) {
  LLVM_DEBUG(dbgs() << "WrapLoop:   Populate byass block\n");

  assert(bbsGen.guard && "Generated guard block");
  assert(bbsGen.bypass && "Generated bypass block");
  assert(succ_size(bbsGen.bypass) &&
         "Generated bypass block has a single successor");

  BasicBlock *guard = bbsGen.guard;
  BasicBlock *bypass = bbsGen.bypass;
  BasicBlock *cont = *succ_begin(bypass);

  Value *syncreg = getWrapperSyncRegion(*guard);
  SyncInst *sync = SyncInst::Create(cont, syncreg);

  ReplaceInstWithInst(bypass->getTerminator(), sync);
}

// Generate loop metadata for the outer loop. This will be a clone of the
// metadata of the loop, \p loop being transformed.
void WrapWithTapirLoop::genWrapperLoopMD(const Loop &loop,
                                         const LoopBlocks &bbsGen) {
  LLVM_DEBUG(dbgs() << "WrapLoop:   Create loop metadata\n");

  assert(loop.getLoopID() && "Loop has ID metadata");
  assert(bbsGen.latch && "Generated wrapper loop latch");

  MDNode *newMD = cloneLoopMetadata(loop);
  BasicBlock *latch = bbsGen.latch;
  Instruction *term = latch->getTerminator();
  term->setMetadata(LLVMContext::MD_loop, newMD);
}

//==------------------------------------------------------------------------==//

bool llvm::checkTapirLoopSafeToWrap(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                                    LoopInfo &li, ScalarEvolution &se) {
  const Loop &loop = *tapirLoop.getLoop();
  Task &task = *tapirLoop.getTask();

  if (!loop.isLoopSimplifyForm())
    return emitDiagnostic(loop, DiagID::ErrLoopNotSimplifyForm);

  if (!getLoopGuardBlock(loop, /*allowDeadEndExits=*/true))
    return emitDiagnostic(loop, DiagID::ErrTapirLoopNoGuardBlock);

  BasicBlock *bbBypass = getLoopBypassBlock(loop, /*allowDeadEndExits=*/true);
  if (!bbBypass)
    return emitDiagnostic(loop, DiagID::ErrTapirLoopNoBypassBlock);
  else if (bbBypass->size() != 1)
    return emitDiagnostic(loop, DiagID::ErrGeneric,
                          "tapir loop bypass block is not empty");
  else if (!isa<SyncInst>(bbBypass->getTerminator()))
    return emitDiagnostic(
        loop, DiagID::ErrGeneric,
        "tapir loop bypass block terminator is not a sync instruction");

  // If the tapir loop induction variable is used outside the loop, we may have
  // problems. Wrapping the loop may result in the definition of the IV not
  // being present on all paths leading to the loop. It is not clear if this can
  // ever happen in practice.
  for (auto &[iv, ivDescr] : *tapirLoop.getInductionVars())
    if (isUsedOutsideLoop(*iv, loop, li))
      return emitDiagnostic(loop, DiagID::ErrTapirLoopIVUsedOutsideLoop);

  // One way to have more than one detach instruction in a tapir loop is if the
  // loop attempts to spawn additional tapir tasks. It is unlikely that we will
  // ever support this, neither here, nor in the frontend. We don't do exactly
  // the same analysis for reattaches because AnalyzeTaskForSerialization will
  // compute the number of reattaches, so we rely on that instead.
  const DetachInst *di = getUniqueInstInLoopOnly<DetachInst>(loop);
  if (!di)
    return emitDiagnostic(loop, DiagID::ErrTapirLoopNoUniqueDetachInst);
  else if (di->getDetached() != task.getEntry())
    return emitDiagnostic(loop, DiagID::ErrTapirLoopTaskEntryMismatch);

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
    return emitDiagnostic(loop, DiagID::ErrTapirLoopThrowsException);

  // Since we allow only a single detach instruction in the tapir loop, we can
  // only have a single reattach.
  if (reattaches.size() != 1)
    return emitDiagnostic(loop, DiagID::ErrTapirLoopNoUniqueReattachInst);

  return true;
}

Loop *llvm::wrapWithSerialLoop(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                               LoopInfo &li, MemorySSA &mssa,
                               ScalarEvolution &se, TaskInfo &ti) {
  Loop *wrapperLoop = WrapWithSerialLoop(dt, li, mssa, se, ti).runOn(tapirLoop);

#ifndef NDEBUG
  assert(!isTapirLoop(*wrapperLoop) && "Wrapper loop is not a tapir loop");
  assert(!getTaskIfTapirLoop(wrapperLoop, &ti) &&
         "Wrapper loop not recognized as tapir loop");
#endif // NDEBUG

  return wrapperLoop;
}

Loop *llvm::wrapWithTapirLoop(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                              LoopInfo &li, MemorySSA &mssa,
                              ScalarEvolution &se, TaskInfo &ti) {
  // getTapirLoopsInNest() performs some sanity checks when collecting the tapir
  // loops. If we collect them after the loop has been wrapped, those checks
  // will fail.
  Loop &root = *tapirLoop.getLoop();
  SmallVector<Loop *, 4> loops = getTapirLoopsInNest(root);

  Loop *wrapperLoop = WrapWithTapirLoop(dt, li, mssa, se, ti).runOn(tapirLoop);

  // The nest depth and the levels of the loops in the nest must be updated.
  // The loops array contain the loops in the original nest. Since this has been
  // wrapped with a tapir loop, the existing level must be incremented by one.
  for (Loop *loop : loops)
    addPerfectLevelAttr(*loop, *getPerfectLevelAttr(*loop) + 1);
  removePerfectDepthAttr(root);
  addPerfectDepthAttr(*wrapperLoop, loops.size() + 1);

#ifndef NDEBUG
  assert(isTapirLoop(*wrapperLoop) && "Wrapper loop is a tapir loop");
  assert(getTaskIfTapirLoop(wrapperLoop, &ti) &&
         "Wrapper loop recognized as tapir loop");
#endif // NDEBUG

  return wrapperLoop;
}

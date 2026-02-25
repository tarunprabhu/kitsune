//===- TapirLoopNestAnalysis.cpp - Analyze nests of tapir loops -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tools to deal with nests of tapir loops.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Analysis/TapirLoopNestAnalysis.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

#define DEBUG_TYPE "tapir-loop-nest-analysis"

using namespace llvm;

static CmpInst *getOuterLoopLatchCmp(const Loop &outerLoop) {
  const BasicBlock *latch = outerLoop.getLoopLatch();
  assert(latch && "Expecting a valid loop latch");

  const auto *br = dyn_cast<BranchInst>(latch->getTerminator());
  assert(br && br->isConditional() &&
         "Loop latch terminator must be a conditional branch instruction");

  return dyn_cast<CmpInst>(br->getCondition());
}

static CmpInst *getInnerLoopGuardCmp(const Loop &innerLoop) {
  if (BranchInst *innerGuard = innerLoop.getLoopGuardBranch())
    if (auto *cmpInst = dyn_cast<CmpInst>(innerGuard->getCondition()))
      return cmpInst;
  return nullptr;
}

static bool checkSafeInstruction(const Instruction &inst,
                                 const CmpInst *innerLoopGuardCmp,
                                 const CmpInst *outerLoopLatchCmp,
                                 const Loop::LoopBounds &outerLoopLB) {

  bool isAllowed = isSafeToSpeculativelyExecute(&inst) || isa<PHINode>(inst) ||
                   isa<BranchInst>(inst) || isa<DetachInst>(inst);
  if (!isAllowed)
    return false;

  // The only binary instruction allowed is the outer loop step instruction,
  // the only comparison instructions allowed are the inner loop guard
  // compare instruction and the outer loop latch compare instruction.
  if (isa<BinaryOperator>(inst) && &inst != &outerLoopLB.getStepInst())
    return false;
  else if (isa<CmpInst>(inst) && &inst != outerLoopLatchCmp &&
           &inst != innerLoopGuardCmp)
    return false;
  return true;
}

/// Check if the given basic block is empty.
static bool isEmpty(const BasicBlock &bb) { return bb.size() == 1; }

/// Check if the given basic block contains a single call to an intrinsic that
/// creates a syncregion.
static bool onlyCreatesSyncRegion(const BasicBlock &bb) {
  if (bb.size() == 2)
    if (const auto *call = dyn_cast<CallInst>(&bb.front()))
      if (Function *f = call->getCalledFunction())
        if (f->getIntrinsicID() == Intrinsic::syncregion_start)
          return true;
  return false;
}

/// Return true if the given basic block is empty, or contains a single call to
/// create a syncregion.
static bool isEmptyOrOnlyCreatesSyncRegion(const BasicBlock &bb) {
  return isEmpty(bb) || onlyCreatesSyncRegion(bb);
}

/// Check if there is a unique path between \p from and \p end. All blocks in
/// this path must be "safe", as determined by the given \p isSafe function.
/// In all cases, the intermediate blocks must have unique successor.
static const BasicBlock *
skipSafeBlocksUntil(const BasicBlock *from, const BasicBlock *end,
                    std::function<bool(const BasicBlock &bb)> isSafe) {
  // Get the unique successor of a basic block. This treats detach instructions
  // as a special case where the detached block is assumed to be the only
  // successor. This is not true in general, but it is true for loops,
  // especially for tapir loops.
  auto getUniqueSuccessor = [](const BasicBlock &bb) -> const BasicBlock * {
    if (const auto *detach = dyn_cast<DetachInst>(bb.getTerminator()))
      return detach->getDetached();
    return bb.getUniqueSuccessor();
  };

  if (from == end || !getUniqueSuccessor(*from))
    return from;

  // `visited` is used to avoid running into an infinite loop.
  SmallPtrSet<const BasicBlock *, 4> visited;
  const BasicBlock *bb = getUniqueSuccessor(*from);
  const BasicBlock *bbPred = from;
  while (bb && bb != end && isSafe(*bb) && !visited.count(bb)) {
    visited.insert(bb);
    bbPred = bb;
    bb = getUniqueSuccessor(*bb);
  }

  return (bb == end) ? end : bbPred;
}

/// Check if there is a unique path between \p from and \p end, where the only
/// basic blocks on the path are "safe" as determined by \p isSafe. Safe blocks
/// are typically empty, or consist exclusively of instructions that may be
/// safely ignored when determining if tapir loops are perfectly nested.
static bool
hasUniqueSafePathBetween(const BasicBlock &from, const BasicBlock &to,
                         std::function<bool(const BasicBlock &)> isSafe) {
  return skipSafeBlocksUntil(&from, &to, isSafe) == &to;
}

/// Determine whether the loops structure violates basic requirements for
/// perfect nesting:
///
///  - the inner loop should be the outer loop's only child
///
///  - the outer loop header should 'flow' into the inner loop preheader
///    or jump around the inner loop to the outer loop latch
///
///  - if the inner loop latch exits the inner loop, it should 'flow' into
///    the outer loop latch.
///
/// Returns true if the loop structure satisfies the basic requirements and
/// false otherwise.
static bool checkLoopsStructure(const Loop &outerLoop, const Loop &innerLoop,
                                ScalarEvolution &se) {
  LLVM_DEBUG(dbgs() << "Checking the structure of loops '"
                    << outerLoop.getName() << "' and '" << innerLoop.getName()
                    << "'.\n";);

  // The inner loop must be the only outer loop's child.
  if (innerLoop.getParentLoop() != &outerLoop) {
    LLVM_DEBUG(dbgs() << "'" << outerLoop.getName() << "' is not a parent of '"
                      << innerLoop.getName() << "'.\n";);
    return false;
  }

  if (outerLoop.getSubLoops().size() != 1) {
    LLVM_DEBUG(dbgs() << "'" << outerLoop.getName()
                      << "' has more than one subloop.\n";);
    return false;
  }

  // We expect loops in normal form which have a preheader, header, latch...
  if (!outerLoop.isLoopSimplifyForm() || !innerLoop.isLoopSimplifyForm()) {
    LLVM_DEBUG(dbgs() << "Both '" << outerLoop.getName() << "' and '"
                      << innerLoop.getName()
                      << "' must be in loop simplify form.\n";);
    return false;
  }

  const BasicBlock *outerLoopHeader = outerLoop.getHeader();
  const BasicBlock *outerLoopLatch = outerLoop.getLoopLatch();
  const BasicBlock *innerLoopPreheader = innerLoop.getLoopPreheader();
  const BasicBlock *innerLoopLatch = innerLoop.getLoopLatch();
  const BasicBlock *innerLoopExit = innerLoop.getExitBlock();

  // We expect rotated loops. The inner loop should have a single exit block.
  if (outerLoop.getExitingBlock() != outerLoopLatch ||
      innerLoop.getExitingBlock() != innerLoopLatch || !innerLoopExit) {
    LLVM_DEBUG(dbgs() << "Both '" << outerLoop.getName() << "' and '"
                      << innerLoop.getName()
                      << "' must be in loop rotate form.\n";);
    return false;
  }

  // Returns whether the block `exitBlock` contains at least one LCSSA Phi node.
  auto containsLCSSAPhi = [](const BasicBlock &exitBlock) {
    return any_of(exitBlock.phis(), [](const PHINode &phi) {
      return phi.getNumIncomingValues() == 1;
    });
  };

  // Returns whether the block `BB` qualifies for being an extra Phi block. The
  // extra Phi block is the additional block inserted after the exit block of an
  // "guarded" inner loop which contains "only" Phi nodes corresponding to the
  // LCSSA Phi nodes in the exit block.
  auto isExtraPhiBlock = [&](const BasicBlock &bb) {
    return &*bb.getFirstNonPHIIt() == bb.getTerminator() &&
           all_of(bb.phis(), [&](const PHINode &phi) {
             return all_of(phi.blocks(), [&](const BasicBlock *incomingBlock) {
               return incomingBlock == innerLoopExit ||
                      incomingBlock == outerLoopHeader;
             });
           });
  };

  /// Returns true if the successor of the from block is the same as the end,
  /// with potential empty or

  const BasicBlock *extraPhiBlock = nullptr;
  // Ensure the only branch that may exist between the loops is the inner loop
  // guard.
  if (outerLoopHeader != innerLoopPreheader) {
    const BasicBlock *outerLoopHeaderSucc = skipSafeBlocksUntil(
        outerLoopHeader, innerLoopPreheader, isEmptyOrOnlyCreatesSyncRegion);

    // no conditional branch present
    if (outerLoopHeaderSucc != innerLoopPreheader) {
      const auto *br =
          dyn_cast<BranchInst>(outerLoopHeaderSucc->getTerminator());

      if (!br || br != innerLoop.getLoopGuardBranch()) {
        LLVM_DEBUG(dbgs() << "Successor of outer loop header must must be "
                             "guard branch of inner loop.\n";);
        return false;
      }

      bool innerLoopExitContainsLCSSA = containsLCSSAPhi(*innerLoopExit);

      // The successors of the inner loop guard should be the inner loop
      // preheader or the outer loop latch possibly through empty blocks.
      for (const BasicBlock *succ : br->successors()) {
        if (hasUniqueSafePathBetween(*succ, *innerLoopPreheader, isEmpty))
          continue;
        if (hasUniqueSafePathBetween(*succ, *outerLoopLatch, isEmpty))
          continue;

        // If `innerLoopExit` contains LCSSA Phi instructions, additional block
        // may be inserted before the `outerLoopLatch` to which `br` jumps. The
        // loops are still considered perfectly nested if the extra block only
        // contains Phi instructions from innerLoopExit and outerLoopHeader.
        if (innerLoopExitContainsLCSSA && isExtraPhiBlock(*succ) &&
            succ->getSingleSuccessor() == outerLoopLatch) {
          // Points to the extra block so that we can reference it later in the
          // final check. We can also conclude that the inner loop is
          // guarded and there exists LCSSA Phi node in the exit block later if
          // we see a non-null `ExtraPhiBlock`.
          extraPhiBlock = succ;
          continue;
        }

        LLVM_DEBUG(dbgs() << "Inner loop guard successor " << succ->getName()
                          << " doesn't lead to inner loop preheader or "
                             "outer loop latch.\n";);
        return false;
      }
    }
  }

  // Ensure the inner loop exit block lead to the outer loop latch possibly
  // through empty blocks.
  if ((!extraPhiBlock ||
       !hasUniqueSafePathBetween(*innerLoopExit, *extraPhiBlock, isEmpty)) &&
      !hasUniqueSafePathBetween(*innerLoopExit, *outerLoopLatch, isEmpty)) {
    LLVM_DEBUG(dbgs() << "Inner loop exit block  does not directly lead to the "
                         "outer loop latch.\n";);
    return false;
  }

  return true;
}

static bool arePerfectlyNested(const Loop &outerLoop, const Loop &innerLoop,
                               ScalarEvolution &se) {
  assert(outerLoop.getSubLoops().size() && "Outer loop should have subloops");
  assert(innerLoop.getParentLoop() && "Inner loop should have a parent");
  LLVM_DEBUG(dbgs() << "Checking whether loop '" << outerLoop.getName()
                    << "' and '" << innerLoop.getName()
                    << "' are perfectly nested.\n");

  if (!checkLoopsStructure(outerLoop, innerLoop, se)) {
    LLVM_DEBUG(dbgs() << "Not perfectly nested: invalid loop structure.\n");
    return false;
  }

  // Bail out if we cannot retrieve the outer loop bounds.
  std::optional<Loop::LoopBounds> outerLoopLB = outerLoop.getBounds(se);
  if (!outerLoopLB) {
    LLVM_DEBUG(dbgs() << "Cannot compute loop bounds of outerLoop: "
                      << outerLoop << "\n";);
    return false;
  }

  CmpInst *outerLoopLatchCmp = getOuterLoopLatchCmp(outerLoop);
  CmpInst *innerLoopGuardCmp = getInnerLoopGuardCmp(innerLoop);

  // Determine whether instructions in a basic block are one of:
  //  - the inner loop guard comparison
  //  - the outer loop latch comparison
  //  - the outer loop induction variable increment
  //  - a phi node, a cast or a branch
  auto containsOnlySafeInstructions = [&](const BasicBlock &bb) {
    return llvm::all_of(bb, [&](const Instruction &inst) {
      return checkSafeInstruction(inst, innerLoopGuardCmp, outerLoopLatchCmp,
                                  *outerLoopLB);
    });
  };

  // Check the code surrounding the inner loop for instructions that are deemed
  // unsafe.
  const BasicBlock *outerLoopHeader = outerLoop.getHeader();
  const BasicBlock *outerLoopLatch = outerLoop.getLoopLatch();
  const BasicBlock *innerLoopPreHeader = innerLoop.getLoopPreheader();

  if (!containsOnlySafeInstructions(*outerLoopHeader) ||
      !containsOnlySafeInstructions(*outerLoopLatch) ||
      (innerLoopPreHeader != outerLoopHeader &&
       !containsOnlySafeInstructions(*innerLoopPreHeader)) ||
      !containsOnlySafeInstructions(*innerLoop.getExitBlock())) {
    LLVM_DEBUG(dbgs() << "Not perfectly nested: code surrounding inner loop is "
                         "unsafe\n";);
    return false;
  }

  LLVM_DEBUG(dbgs() << "Loop '" << outerLoop.getName() << "' and '"
                    << innerLoop.getName() << "' are perfectly nested.\n");
  return true;
}

TapirLoopNest::TapirLoopNest(Loop &loop, TaskInfo &ti, ScalarEvolution &se)
    : nest(loop, se) {
  unsigned outermostDepth = nest.getOutermostLoop().getLoopDepth();
  unsigned depth = nest.getNestDepth();
  // `loop` is guaranteed to be a tapir loop. It is perfect by definition. At
  // each level, we expect exactly one tapir loop if it is to be a perfect
  // tapir loop nest.
  perfectTapirLoops.push_back(&loop);
  for (unsigned d = outermostDepth + 1; d < outermostDepth + depth; ++d) {
    LoopVectorTy loops = nest.getLoopsAtDepth(d);
    assert(!loops.empty() && "Loops at given depth not found");

    Loop *outerLoop = perfectTapirLoops.back();
    Loop *loop = loops.front();
    if (!getTaskIfTapirLoop(loop, &ti) ||
        !arePerfectlyNested(*outerLoop, *loop, se))
      break;
    perfectTapirLoops.push_back(loop);
  }
}

std::unique_ptr<TapirLoopNest> TapirLoopNest::create(Loop &loop, TaskInfo &ti,
                                                     ScalarEvolution &se) {
  if (!getTaskIfTapirLoop(&loop, &ti))
    return nullptr;

  return std::unique_ptr<TapirLoopNest>(new TapirLoopNest(loop, ti, se));
}

bool llvm::isTapirLoop(Loop &loop, TaskInfo &ti) {
  return getTaskIfTapirLoop(&loop, &ti);
}

// Return true if any of the ancestors of a loop are tapir loops. The given
// loop is not required to be a tapir loop. If the given loop is a top-level
// loop, return false.
static bool isAnyAncestorTapirLoop(Loop &loop, TaskInfo &ti) {
  Loop *parentLoop = loop.getParentLoop();
  if (!parentLoop)
    return false;
  else if (isTapirLoop(*parentLoop, ti))
    return true;
  else
    return isAnyAncestorTapirLoop(*parentLoop, ti);
}

bool llvm::isTopLevelTapirLoop(Loop &loop, TaskInfo &ti) {
  return isTapirLoop(loop, ti) && not isAnyAncestorTapirLoop(loop, ti);
}

bool llvm::isTapirLoopForGPU(Loop &loop, TaskInfo &ti) {
  if (!isTapirLoop(loop, ti))
    return false;

  TTID tt = *getTapirLoopTargetAttr(loop);
  if (tt != TTID::Cuda && tt != TTID::Hip)
    return false;

  for (Loop *subLoop : getAllSubLoops(loop))
    if (isTapirLoop(*subLoop, ti))
      if (getTapirLoopTargetAttr(*subLoop) != tt)
        return false;

  return true;
}

bool llvm::isTopLevelTapirLoopForGPU(Loop &loop, TaskInfo &ti) {
  return isTopLevelTapirLoop(loop, ti) && isTapirLoopForGPU(loop, ti);
}

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
#include "kitsune/Core/InstUtils.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

#define DEBUG_TYPE "tapir-loop-nest-analysis"

using namespace llvm;

namespace {

/// Check if a given pair of loops are perfectly nested with respect to one
/// another.
class PerfectNestChecker {
protected:
  using IsSafeBB = std::function<bool(PerfectNestChecker *, BasicBlock &)>;
  using IsSafeInst = std::function<bool(Instruction &)>;

private:
  /// The instructions whose presence make the loop nest imperfect.
  SmallSetVector<Instruction *, 4> &unsafeInsts;

protected:
  /// Check if the given basic block is empty.
  bool isEmpty(BasicBlock &bb);
  bool isEmptyRecordUnsafe(BasicBlock &bb);

  /// Return true if the given basic block is empty, or contains a single call
  /// to create a syncregion.
  bool isEmptyOrOnlyCallsSyncRegionStart(BasicBlock &bb);

  /// Check that all instructions in the given block are safe as determined by
  /// \p isInstSafe.
  bool checkInstsInBlock(BasicBlock &bb,
                         std::function<bool(Instruction &)> isInstSafe,
                         StringRef label);

  /// Check if there is a unique path between \p from and \p end. All blocks in
  /// this path must be "safe", as determined by the given \p isSafe function.
  /// In all cases, the intermediate blocks must have unique successor.
  BasicBlock *skipSafeBlocksUntil(BasicBlock *from, BasicBlock *end,
                                  IsSafeBB isSafeBB);

  /// Check if there is a unique path between \p from and \p end, where the only
  /// basic blocks on the path are "safe" as determined by \p isSafe. Safe
  /// blocks are typically empty, or consist exclusively of instructions that
  /// may be safely ignored when determining if tapir loops are perfectly
  /// nested.
  bool hasUniqueSafePathBetween(BasicBlock &from, BasicBlock &to,
                                IsSafeBB isSafeBB);

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
  bool checkLoopsStructure(Loop &outerLoop, Loop &innerLoop,
                           ScalarEvolution &se);

  /// Get the compare instruction that guards the inner loop. This will skip
  /// the body of the loop if the trip count is less than the start value of the
  /// loop induction variable.
  CmpInst *getInnerLoopGuardCmp(Loop &innerLoop) const;

  /// Check if the outer loop header only contains the expected set of
  /// instructions. In addition to a certain set of instructions, the outer loop
  /// header may contain the inner loop guard branch, i.e. the branch that skips
  /// the inner loop entirely if the loop trip count is determined to be <= 0.
  bool checkOuterLoopHeader(BasicBlock &header, CmpInst *innerLoopGuardCmp);

  bool checkOuterLoopLatch(BasicBlock &latch, CmpInst *latchCmpInst,
                           const Loop::LoopBounds &bounds);

public:
  PerfectNestChecker(SmallSetVector<Instruction *, 4> &unsafeInsts);

  bool run(Loop &outerLoop, Loop &innerLoop, ScalarEvolution &se);
};

} // namespace

PerfectNestChecker::PerfectNestChecker(
    SmallSetVector<Instruction *, 4> &unsafeInsts)
    : unsafeInsts(unsafeInsts) {}

bool PerfectNestChecker::isEmptyRecordUnsafe(BasicBlock &bb) {
  if (bb.size() == 1)
    return true;

  for (Instruction &inst : bb)
    if (!inst.isTerminator())
      unsafeInsts.insert(&inst);
  return false;
}

bool PerfectNestChecker::isEmpty(BasicBlock &bb) { return bb.size() == 1; }

bool PerfectNestChecker::isEmptyOrOnlyCallsSyncRegionStart(BasicBlock &bb) {
  auto onlyCallsSyncRegionStart = [&](BasicBlock &bb) -> bool {
    return bb.size() == 2 && isCallSyncRegionStart(bb.front());
  };

  // We shouldn't call isEmpty() here, or we will end up recording unsafe
  // instructions multiple times.
  if (isEmpty(bb) || onlyCallsSyncRegionStart(bb))
    return true;

  for (Instruction &inst : bb)
    if (!inst.isTerminator() && !isCallSyncRegionStart(inst))
      unsafeInsts.insert(&inst);
  return false;
}

BasicBlock *PerfectNestChecker::skipSafeBlocksUntil(BasicBlock *from,
                                                    BasicBlock *end,
                                                    IsSafeBB isSafeBB) {
  // Get the unique successor of a basic block. This treats detach instructions
  // as a special case where the detached block is assumed to be the only
  // successor. This is not true in general, but it is true for loops,
  // especially for tapir loops.
  auto getUniqueSuccessor = [](BasicBlock &bb) -> BasicBlock * {
    if (auto *detach = dyn_cast<DetachInst>(bb.getTerminator()))
      return detach->getDetached();
    return bb.getUniqueSuccessor();
  };

  if (from == end || !getUniqueSuccessor(*from))
    return from;

  // `visited` is used to avoid running into an infinite loop.
  SmallPtrSet<BasicBlock *, 4> visited;
  BasicBlock *bb = getUniqueSuccessor(*from);
  BasicBlock *bbPred = from;
  while (bb && bb != end && isSafeBB(this, *bb) && !visited.count(bb)) {
    visited.insert(bb);
    bbPred = bb;
    bb = getUniqueSuccessor(*bb);
  }

  return (bb == end) ? end : bbPred;
}

bool PerfectNestChecker::hasUniqueSafePathBetween(BasicBlock &from,
                                                  BasicBlock &to,
                                                  IsSafeBB isSafeBB) {
  return skipSafeBlocksUntil(&from, &to, isSafeBB) == &to;
}

bool PerfectNestChecker::checkLoopsStructure(Loop &outerLoop, Loop &innerLoop,
                                             ScalarEvolution &se) {
  LLVM_DEBUG(dbgs() << "Checking the structure of loops '"
                    << outerLoop.getName() << "' and '" << innerLoop.getName()
                    << "'.\n";);

  BasicBlock *outerLoopHeader = outerLoop.getHeader();
  BasicBlock *outerLoopLatch = outerLoop.getLoopLatch();
  BasicBlock *innerLoopPreheader = innerLoop.getLoopPreheader();
  BasicBlock *innerLoopLatch = innerLoop.getLoopLatch();
  BasicBlock *innerLoopExit = innerLoop.getExitBlock();

  // We expect rotated loops. The inner loop should have a single exit block.
  if (outerLoop.getExitingBlock() != outerLoopLatch ||
      innerLoop.getExitingBlock() != innerLoopLatch || !innerLoopExit) {
    LLVM_DEBUG(dbgs() << "Both '" << outerLoop.getName() << "' and '"
                      << innerLoop.getName()
                      << "' must be in loop rotate form.\n";);
    return false;
  }

  // Returns whether the block `exitBlock` contains at least one LCSSA Phi node.
  auto containsLCSSAPhi = [](BasicBlock &exitBlock) {
    return any_of(exitBlock.phis(),
                  [](PHINode &phi) { return phi.getNumIncomingValues() == 1; });
  };

  // Returns whether the block `bb` qualifies for being an extra Phi block. The
  // extra Phi block is the additional block inserted after the exit block of an
  // "guarded" inner loop which contains "only" Phi nodes corresponding to the
  // LCSSA Phi nodes in the exit block.
  auto isExtraPhiBlock = [&](BasicBlock &bb) {
    return &*bb.getFirstNonPHIIt() == bb.getTerminator() &&
           all_of(bb.phis(), [&](PHINode &phi) {
             return all_of(phi.blocks(), [&](BasicBlock *incomingBlock) {
               return incomingBlock == innerLoopExit ||
                      incomingBlock == outerLoopHeader;
             });
           });
  };

  IsSafeBB isEmpty = &PerfectNestChecker::isEmpty;

  // Ensure the only branch that may exist between the loops is the inner loop
  // guard.
  BasicBlock *extraPhiBlock = nullptr;
  if (outerLoopHeader != innerLoopPreheader) {
    BasicBlock *outerLoopHeaderSucc = skipSafeBlocksUntil(
        outerLoopHeader, innerLoopPreheader,
        &PerfectNestChecker::isEmptyOrOnlyCallsSyncRegionStart);

    // no conditional branch present
    if (outerLoopHeaderSucc != innerLoopPreheader) {
      auto *br = dyn_cast<BranchInst>(outerLoopHeaderSucc->getTerminator());

      if (!br || br != innerLoop.getLoopGuardBranch()) {
        LLVM_DEBUG(dbgs() << "Successor of outer loop header must must be "
                             "guard branch of inner loop.\n";);
        return false;
      }

      bool innerLoopExitContainsLCSSA = containsLCSSAPhi(*innerLoopExit);

      // The successors of the inner loop guard should be the inner loop
      // preheader or the outer loop latch possibly through empty blocks.
      for (BasicBlock *succ : br->successors()) {
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

CmpInst *PerfectNestChecker::getInnerLoopGuardCmp(Loop &innerLoop) const {
  if (BranchInst *innerGuard = innerLoop.getLoopGuardBranch())
    if (auto *cmpInst = dyn_cast<CmpInst>(innerGuard->getCondition()))
      return cmpInst;
  return nullptr;
}

bool PerfectNestChecker::checkInstsInBlock(BasicBlock &bb,
                                           IsSafeInst isSafeInst,
                                           StringRef label) {
  for (Instruction &inst : bb) {
    if (!isSafeInst(inst)) {
      LLVM_DEBUG(dbgs() << "Unsafe instruction in " << label << ": " << inst
                        << "\n");
      return false;
    }
  }
  return true;
}

bool PerfectNestChecker::checkOuterLoopHeader(BasicBlock &header,
                                              CmpInst *innerLoopGuardCmp) {
  auto isSafeInst = [&innerLoopGuardCmp](Instruction &inst) -> bool {
    // The only comparison instruction allowed is the inner loop guard
    // comparison. Otherwise, PHINode's, BranchInst's and DetachInst's are
    // allowed, though we should check that these are exactly those that we
    // expect.
    if (isa<CmpInst>(inst))
      return &inst == innerLoopGuardCmp;
    else if (isa<PHINode>(inst) || isa<BranchInst>(inst) ||
             isa<DetachInst>(inst))
      return true;
    return false;
  };

  return checkInstsInBlock(header, isSafeInst, "outer loop header");
}

bool PerfectNestChecker::checkOuterLoopLatch(BasicBlock &latch,
                                             CmpInst *latchCmpInst,
                                             const Loop::LoopBounds &bounds) {
  Instruction *step = &bounds.getStepInst();

  auto isSafeInst = [&latchCmpInst, &step](Instruction &inst) -> bool {
    // The only binary instruction allowed is the outer loop step instruction,
    // the only comparison instruction allowed is the outer loop latch compare
    // instruction. Otherwise, certain instructions are safe, but nothing else
    // is.
    if (isa<CmpInst>(inst))
      return &inst == latchCmpInst;
    else if (isa<BinaryOperator>(inst))
      return &inst == step;
    else if (isa<BranchInst>(inst))
      return true;
    return false;
  };

  return checkInstsInBlock(latch, isSafeInst, "outer loop latch");
}

bool PerfectNestChecker::run(Loop &outerLoop, Loop &innerLoop,
                             ScalarEvolution &se) {
  LLVM_DEBUG(dbgs() << "Checking whether loop '" << outerLoop.getName()
                    << "' and '" << innerLoop.getName()
                    << "' are perfectly nested.\n");

  if (!checkLoopsStructure(outerLoop, innerLoop, se)) {
    LLVM_DEBUG(dbgs() << "Not perfectly nested: invalid loop structure.\n");
    return false;
  }

  // Check the code surrounding the inner loop for instructions that are deemed
  // unsafe.
  BasicBlock *outerHeader = outerLoop.getHeader();
  BasicBlock *outerLatch = outerLoop.getLoopLatch();
  CmpInst *outerLatchCmp = outerLoop.getLatchCmpInst();
  const std::optional<Loop::LoopBounds> outerBounds = outerLoop.getBounds(se);

  BasicBlock *innerPreheader = innerLoop.getLoopPreheader();
  CmpInst *innerGuardCmp = getInnerLoopGuardCmp(innerLoop);

  bool isSafe = checkOuterLoopHeader(*outerHeader, innerGuardCmp) &&
                checkOuterLoopLatch(*outerLatch, outerLatchCmp, *outerBounds);
  if (innerPreheader != outerHeader) {
    // TODO: In this case, we expect that the inner loop exit block is
    // terminated with a sync instruction. If the preheader contains a call to
    // start a syncregion, it must be the one that is associated with the inner
    // loop. These should be checked here, just to be safe. For now, we are
    // relying on Kitsune's verifier running and bailing out with an error if
    // the tapir loops are not structured exactly as we expect.
    BasicBlock &innerExit = *innerLoop.getExitBlock();
    isSafe &= isEmptyOrOnlyCallsSyncRegionStart(*innerPreheader) &&
              isEmptyRecordUnsafe(innerExit);
  }

  if (!isSafe) {
    LLVM_DEBUG(
        dbgs() << "Not perfectly nested: code surrounding inner loop is unsafe"
               << "\n";);
    return false;
  }

  LLVM_DEBUG(dbgs() << "Loop '" << outerLoop.getName() << "' and '"
                    << innerLoop.getName() << "' are perfectly nested.\n");
  return true;
}

static bool checkLoopSimplifyForm(const Loop &loop) {
  if (!loop.isLoopSimplifyForm()) {
    LLVM_DEBUG(dbgs() << "'" << loop.getName() << "', at depth "
                      << loop.getLoopDepth()
                      << "is not in loop-simplify form.\n";);
    return false;
  }
  return true;
}

// Sanity check an outer loop. This is just an outerLoop relative to some other
// "inner" loop. It need not be the outermost loop in a nest. Return false if at
// least one check fails, true otherwise.
bool TapirLoopNest::sanityCheckOuterLoop(const Loop &loop,
                                         ScalarEvolution &se) const {
  if (!checkLoopSimplifyForm(loop))
    return false;

  unsigned depth = loop.getLoopDepth();
  LoopVectorTy subLoops = nest.getLoopsAtDepth(depth + 1);
  if (subLoops.size() != 1) {
    LLVM_DEBUG(dbgs() << "'" << loop.getName() << "' at depth " << depth
                      << "' has more than one subloop.\n";);
    return false;
  }

  if (!loop.getBounds(se)) {
    LLVM_DEBUG(dbgs() << "Cannot compute loop bounds of loop '"
                      << loop.getName() << "' at depth " << depth << "\n";);
    return false;
  }

  return true;
}

// Sanity check an inner loop. Return false if at least one check fails, true
// otherwise.
bool TapirLoopNest::sanityCheckInnerLoop(const Loop &loop,
                                         ScalarEvolution &se) const {
  if (!checkLoopSimplifyForm(loop))
    return false;
  return true;
}

TapirLoopNest::TapirLoopNest(Loop &root, ScalarEvolution &se) : nest(root, se) {
  assert(isTapirLoop(root) && "Root of tapir loop nest must be a tapir loop");
  assert(root.isLoopSimplifyForm() &&
         "Root of tapir loop nest must be in loop simplify form");
  assert(root.isRotatedForm() &&
         "Root of tapir loop nest must be in loop rotate form");
  LLVM_DEBUG(dbgs() << "Outermost loop: " << getName(root) << "\n");

  // `root` is guaranteed to be a tapir loop. It is perfect by definition.
  perfectTapirLoops.push_back(&root);

  // The depth of the outermost loop is not guaranteed to be 1.
  unsigned outermostDepth = nest.getOutermostLoop().getLoopDepth();
  unsigned depth = nest.getNestDepth();
  for (unsigned d = outermostDepth + 1; d < outermostDepth + depth; ++d) {
    Loop *outerLoop = perfectTapirLoops.back();
    LLVM_DEBUG(dbgs() << "Checking loop at depth " << d << ": "
                      << getName(*outerLoop) << "\n");

    if (!sanityCheckOuterLoop(*outerLoop, se)) {
      LLVM_DEBUG(dbgs() << "Outer loop failed sanity check: "
                        << getName(*outerLoop) << "\n");
      break;
    }

    Loop *innerLoop = nest.getLoopsAtDepth(d).front();
    if (!sanityCheckInnerLoop(*innerLoop, se)) {
      LLVM_DEBUG(dbgs() << "Inner loop failed sanity check: "
                        << getName(*innerLoop) << "\n");
      break;
    }

    if (!isTapirLoop(*innerLoop)) {
      LLVM_DEBUG(dbgs() << "Inner loop is not tapir loop: "
                        << getName(*innerLoop) << "\n");
      break;
    }

    PerfectNestChecker perfectNestChecker(unsafeInsts);
    if (!perfectNestChecker.run(*outerLoop, *innerLoop, se)) {
      LLVM_DEBUG(dbgs() << "Inner loop not perfectly nested: "
                        << getName(*innerLoop) << "\n");
      break;
    }

    perfectTapirLoops.push_back(innerLoop);
  }
}

std::unique_ptr<TapirLoopNest> TapirLoopNest::create(Loop &loop,
                                                     ScalarEvolution &se) {
  if (!isTapirLoop(loop)) {
    LLVM_DEBUG(dbgs() << "Root of loop nest, '" << loop.getName()
                      << "', is not a tapir loop.\n");
    return nullptr;
  }
  return std::unique_ptr<TapirLoopNest>(new TapirLoopNest(loop, se));
}

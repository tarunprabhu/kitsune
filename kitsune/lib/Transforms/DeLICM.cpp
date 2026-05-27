//===- DeLICM.cpp - Pass that is the inverse of LICM ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass that is the inverse of the LICM pass.
//
// The current implementation only DeLICM's instructions that are deemed to be
// "unsafe" by the tapir target analysis pass. "Unsafe" instructions are those
// that result in a pair of tapir loops not being perfectly nested relative to
// one another. At the time of writing, there are several limitations in the
// implementation:
//
//   - This will only work on nests of tapir loops of depth 2 or 3 where all
//     loops have the same GPU-centric tapir target i.e. either 'cuda' or 'hip'.
//     In particular, it will only DeLICM instructions present in the loops at
//     depths 1 and 2. It will never move instructions that are outside the
//     outermost loop, i.e. the loop at depth 1, into that loop.
//
//   - For any unsafe instruction to be DeLICM'ed, *all* unsafe instructions
//     must belong to the same basic block. This is to avoid trying to DeLICM
//     "regions" of code.
//
// Clearly, one consequence of the way this pass has been implemented is that it
// may DeLICM instructions that were never hoisted out of a loop in the first
// place - presumably because the programmer wrote them outside of the loop. In
// the future, this pass may be tweaked to hew closer to the intent of the
// programmer by not DeLICM'ing such instructions.
//
// With better cost analysis, this pass could also operate on "outermost" loops
// by DeLICM'ing instructions that are outside the outermost loop (this will
// also have the effect of enabling DeLICM on loops of depth 1).
//
//
// ------------------------- MAJOR ISSUES TO BE FIXED -------------------------
//
// - This implementation will DeLICM unsafe instructions regardless of their
//   effect on program semantics. In particular, function calls will also be
//   DeLICM'ed. This is a *VERY BAD* idea in general and will be removed very
//   soon. This decision was made _solely_ in the interest of getting an
//   implementation out the door.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/DeLICM.h"
#include "kitsune/Analysis/TapirLoopNestAnalysis.h"
#include "kitsune/Core/InstAttrs.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/TTUtils.h"
#include "kitsune/Passes/PassUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"

#define DEBUG_TYPE "kit-delicm"

using namespace llvm;

namespace {

class DeLICM {
public:
  using UnsafeInstsList = TapirLoopNest::UnsafeInstsList;
  using SmallBBSetVector = SmallSetVector<BasicBlock *, 4>;

private:
  DominatorTree &dt;
  LoopInfo &li;
  ScalarEvolution &se;

private:
  /// If a tapir loop nest could be potentially be perfect, compute what that
  /// nest's depth would be. For instance, in the nest below, if <code> could be
  /// sunk into `forall (k ...)`, this should return all three loops in the
  /// nest.
  ///
  /// \code
  ///   forall (i ...)         // loop at level 1
  ///     forall (j ...)       // loop at level 2
  ///       <code>
  ///       forall (k ...)     // loop at level 3
  /// \endcode
  ///
  /// However, for the loop nest below, this will return only the outermost loop
  /// because the presence of the j2 loop implies that the maximum perfect nest
  /// depth can be at most 1.
  ///
  /// \code
  ///   forall (i ...)         // loop at level 1 and root of the nest
  ///     forall (j1 ...)      // loop at level 2
  ///       forall (k ...)     // loop at level 3
  ///     forall (j2 ...)      // loop at level 2
  /// \endcode
  ///
  /// Similarly, for the loop nest below, this will also return only the
  /// outermost loop, though in this case it is due to the presence of the
  /// non-tapir loop at depth 2.
  ///
  /// \code
  ///   forall (i ...)         // loop at level 1
  ///     for (j ...)          // loop at level 2
  ///       forall (k ...)     // loop at level 3
  /// \endcode
  ///
  SmallVector<Loop *, 4> getPotentialPerfectTapirLoops(Loop &root) const;

  /// Should we attempt to run DeLICM on the loop nest consisting of loops in
  /// \p loopNest. The loop at element `i` in the \p loopNest vector must be the
  /// only loop at depth `i` in the loop nest rooted at the loop at index 0 of
  /// \p loopNest. All loops in \p loopNest must be tapir loops.
  ///
  /// Currently, all loops in \p loopNest must be tapir loops to be lowered
  /// using a GPU-centric tapir target. In the future, this implementation may
  /// be changed.
  bool shouldDeLICM(const SmallVectorImpl<Loop *> &loopNest) const;

  /// Given a list of "unsafe" instructions i.e. instructions that result in
  /// a pair of loops not being imperfectly nested relative to one another,
  /// should we attempt to sink the instructions into the body of the inner
  /// loop.
  ///
  /// Currently, this will only return true if all the instructions are in the
  /// same basic block. This is because we cannot yet deal with sinking regions
  /// of code, SESE (Single-Entry-Single-Exit) or otherwise. We also do not
  /// support sinking a subset of the unsafe instructions.
  ///
  /// NOTE: While it is unlikely performing DeLICM on regions will ever be
  /// supported (though it would be interesting to do so), we may eventually
  /// support DeLICM on a subset of unsafe instructions. In that case, we should
  /// remove this function with another with an appropriate name and signature.
  bool shouldDeLICM(const UnsafeInstsList &unsafeInsts) const;

  /// Are all the given instructions in the same basic block.
  bool allInstsInSameBlock(const UnsafeInstsList &insts) const;

  /// Get the list of basic blocks containing instructions that use the given
  /// instruction.
  SmallBBSetVector getParentsOfUses(Instruction &inst) const;

  /// Get a suitable insertion point, within the basic block \bb, for the
  /// instruction \p inst.
  BasicBlock::iterator getInsertionPointFor(Instruction *inst) const;

  /// Move the given unsafe instructions. These are instructions that resulted
  /// in the only child of \p loop not being perfectly nested relative to
  /// \p loop. The instructions are moved as deep as possible into the body of
  /// the only child of loop. Return true if at least one instruction was moved,
  /// false otherwise.
  bool moveUnsafeInsts(const UnsafeInstsList &unsafeInsts, Loop &loop);

public:
  DeLICM(DominatorTree &dt, LoopInfo &li, ScalarEvolution &se)
      : dt(dt), li(li), se(se) {}

  bool run(Function &f);
};

} // namespace

SmallVector<Loop *, 4> DeLICM::getPotentialPerfectTapirLoops(Loop &root) const {
  LoopNest nest(root, se);
  SmallVector<Loop *, 4> tapirLoops = {&root};
  unsigned depthRoot = root.getLoopDepth();
  unsigned depthInner = depthRoot + nest.getNestDepth();
  for (unsigned d = depthRoot + 1; d < depthInner; ++d) {
    LoopVectorTy loops = nest.getLoopsAtDepth(d);
    if (loops.size() != 1)
      break;

    Loop *loop = loops.front();
    if (!isTapirLoop(*loop))
      break;

    tapirLoops.push_back(loop);
  }
  return tapirLoops;
}

bool DeLICM::shouldDeLICM(const SmallVectorImpl<Loop *> &loopNest) const {
  if (loopNest.size() == 1 || loopNest.size() > 3)
    return false;
  for (const Loop *loop : loopNest)
    if (!isGPUTT(*getTargetAttr(*loop)))
      return false;
  return true;
}

bool DeLICM::shouldDeLICM(const UnsafeInstsList &insts) const {
  if (insts.size()) {
    BasicBlock *bb = insts.front()->getParent();
    for (Instruction *inst : insts)
      if (inst->getParent() != bb)
        return false;
  }
  return true;
}

DeLICM::SmallBBSetVector DeLICM::getParentsOfUses(Instruction &inst) const {
  SmallBBSetVector bbs;
  for (User *u : inst.users()) {
    auto *userInst = dyn_cast<Instruction>(u);
    assert(userInst && "All users of instruction must be instructions");
    bbs.insert(userInst->getParent());
  }
  return bbs;
}

BasicBlock::iterator DeLICM::getInsertionPointFor(Instruction *inst) const {
  // First, collect the basic blocks containing the uses of the instruction.
  // We pick the nearest common dominator of the blocks containing uses of the
  // instruction. This ensures that the instruction will be "executed" as "late"
  // as possible before its value is needed. If the instruction has any uses in
  // this block, we set the insertion point to be immediately before the first
  // use of the instruction in the block. Otherwise, it is inserted as early
  // as possible in the block.
  SmallBBSetVector bbs = getParentsOfUses(*inst);
  BasicBlock *dest = dt.findNearestCommonDominator(iterator_range(bbs));

  for (User *u : inst->users())
    if (auto *userInst = dyn_cast<Instruction>(u))
      if (userInst->getParent() == dest)
        return userInst->getIterator();
  return dest->getFirstInsertionPt();
}

bool DeLICM::moveUnsafeInsts(const UnsafeInstsList &unsafeInsts, Loop &loop) {
  bool changed = false;
  bool moved;
  do {
    moved = false;
    for (Instruction *inst : unsafeInsts) {
      LLVM_DEBUG(dbgs() << "DeLICM: unsafe: " << *inst << "\n");
      BasicBlock::iterator dest = getInsertionPointFor(inst);

      // In some cases, the best place for an instruction might be in the same
      // loop. In this case, we should not move it.
      bool movingToSameLoop = li.getLoopFor(dest->getParent()) == &loop;

      // Since this is looping over the same set of instructions until
      // convergence, we must ensure that we do not "move" an instruction to
      // exactly where it already is. This can happen after the first round of
      // moves.
      bool movingToSamePoint = dest->getPrevNode() == inst;

      if (!movingToSameLoop && !movingToSamePoint) {
        LLVM_DEBUG(dbgs() << "DeLICM: move before: " << *dest << "\n");
        inst->moveBeforePreserving(dest);
        moved = true;
        changed = true;
      }
    }
  } while (moved);

  return changed;
}

bool DeLICM::run(Function &f) {
  bool changed = false;
  for (Loop *root : getTopLevelTapirLoops(li)) {
    SmallVector<Loop *, 4> loops = getPotentialPerfectTapirLoops(*root);
    if (shouldDeLICM(loops)) {
      for (unsigned d = 0; d < loops.size() - 1; ++d) {
        // At this point, we know that loops.size is in [2,3]. We also know that
        // each loop has exactly one child. For each loop, `l` in the nest, we
        // construct a tapir loop nest object. If the max. perfect depth of that
        // nest is 1, it implies that the only child of `l` is not perfectly
        // nested relative to `l`. If this imperfect nesting is due to the
        // presence of "unsafe" instructions i.e. instructions, it may be
        // possible to push those instructions down to the body of an inner
        // loop.
        Loop *loop = loops[d];
        std::unique_ptr<TapirLoopNest> nest = TapirLoopNest::create(*loop, se);
        const UnsafeInstsList &unsafeInsts = nest->getUnsafeInsts();
        if (nest->getMaxPerfectDepth() == 1 && shouldDeLICM(unsafeInsts)) {
          LLVM_DEBUG(dbgs() << "DeLICM: In loop: " << getName(*loop) << "\n");
          changed |= moveUnsafeInsts(unsafeInsts, *loop);
        }
      }
    }
  }

  return changed;
}

PreservedAnalyses DeLICMPass::run(Function &f, FunctionAnalysisManager &am) {
  DominatorTree &dt = am.getResult<DominatorTreeAnalysis>(f);
  LoopInfo &li = am.getResult<LoopAnalysis>(f);
  ScalarEvolution &se = am.getResult<ScalarEvolutionAnalysis>(f);

  bool changed = DeLICM(dt, li, se).run(f);
  if (changed)
    return getPreservedAnalysesCFG();
  return PreservedAnalyses::all();
}

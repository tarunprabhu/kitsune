//===- GuardLoop.cpp - Ensure that a loop is guarded ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Ensure that a tapir loop is guarded. Consider the loop shown below
//
//     ph:
//       ...
//       label %header
//
//     header:
//     latch:
//       br ..., label %exit, label %header
//
//     exit:
//       ...
//
// Here, the loop body is guaranteed to execute at least once. The ellipses in
// the preheader (ph) and exit (exit) blocks indicate that there may be
// instructions in those blocks. This pass transforms this to the following:
//
//     ph:
//       ...
//       br i1 true, label %guard.ph, label %exit
//
//     guard.ph:
//       br label %header
//
//     header:
//     latch:
//       br ..., label %guard.loopexit, label %header
//
//     guard.loopexit:
//       br label %exit
//
//     exit:
//       ...
//
// Note that the instructions in the original loop preheader, ph, are preserved
// before the added guard branch is evaluated, while the instructions in the
// original loop exit branch, exit, are added to the loop bypass block. If a
// later pass were to change the condition in the guard branch instruction,
// those instructions would always be executed, even if the loop is not
// executed.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/GuardLoop.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Passes/PassUtils.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/IR/Constants.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

namespace {
class InsertGuardBlock {
protected:
  DominatorTree &dt;
  LoopInfo &li;

  DomTreeUpdater dtu;
  MemorySSAUpdater mssau;

protected:
public:
  InsertGuardBlock(DominatorTree &dt, LoopInfo &li, MemorySSA &mssa)
      : dt(dt), li(li), dtu(dt, DomTreeUpdater::UpdateStrategy::Eager),
        mssau(&mssa) {}

  bool runOn(Loop &loop);
};

} // namespace

bool InsertGuardBlock::runOn(Loop &loop) {
  assert(isTapirLoop(loop) && "Loop is a tapir loop");
  assert(!getLoopGuardBlock(loop, /*allowDeadEndExits=*/true) &&
         "Loop is not guarded");
  assert(loop.getLoopPreheader() && "Loop has a preheader");
  assert(getUniqueNonDeadEndExitBlock(loop) &&
         "Loop has a unique non-deadend exit block");

  LLVMContext &ctx = getContext(loop);
  Type *i1 = Type::getInt1Ty(ctx);

  BasicBlock *ph = loop.getLoopPreheader();
  BasicBlock *exit = getUniqueNonDeadEndExitBlock(loop);

  // The kit-prepare pass may modify the loop guard so the body is conditionally
  // executed. By splitting the preheader at the terminator, and adding the
  // guard instruction to the end of the original block, we ensure that the
  // instructions in the preheader are always executed, even if any updated
  // guard condition later causes the loop to be skipped.
  BasicBlock *newPh =
      SplitBlock(ph, ph->getTerminator(), &dtu, &li, &mssau, "guard.ph");

  // We split the loop exit block at the beginning, so the new loop exit block
  // is empty. All instructions that were originally in the loop exit block will
  // now be in the bypass block. This way, if the loop guard condition is later
  // updated, and the loop skipped at runtime, the original instructions in the
  // loop exit block will, nevertheless always be executed.
  (void)splitBlockBefore(exit, exit->begin(), &dtu, &li, &mssau,
                         "guard.loopexit");

  // We use splitBlockBefore so the new empty block will get the name
  // guard.loopexit. This is purely to make it easier to read IR should we need
  // to. After this is executed, exit will actually be the bypass block.
  BasicBlock *bypass = exit;

  // At this point, newPh will be an empty block with an unconditional branch
  // to the loop header. All the instructions that were in the original
  // preheader will remain in ph, which must become the new loop guard block.
  // That will happen when we replace its terminator with an appropriate
  // conditional branch.
  Constant *t = ConstantInt::getTrue(i1);
  ReplaceInstWithInst(ph->getTerminator(),
                      CondBrInst::Create(t, newPh, bypass));

  // Now, we can update the analyses.
  dt.insertEdge(ph, bypass);

#ifndef NDEBUG
  dt.verify();
  li.verify(dt);
  loop.verifyLoop();

  assert(loop.isLoopSimplifyForm() &&
         "Loop is in simplify form after a guard has been added");
  assert(getLoopGuardBlock(loop, /*allowDeadEndExits=*/true) == ph &&
         "Got the expected loop guard block");
  assert(getLoopBypassBlock(loop, /*allowDeadEndExits=*/true) == bypass &&
         "Got the expected loop bypass block");
#endif // NDEBUG

  return true;
}

static bool check(const Loop &loop) {
  if (!loop.isLoopSimplifyForm())
    return emitDiagnostic(loop, DiagID::ErrLoopNotSimplifyForm);

  // Strictly speaking, this pass does not require the loop to have a
  // canonical induction variable. However, in Kitsune's pass pipeline, this is
  // run immediately after IndVarSimplify, so we should have one, and we might
  // as well check for it since its presence may make this implementation more
  // robust.
  if (!loop.getCanonicalInductionVariable())
    return emitDiagnostic(loop, DiagID::ErrTapirLoopIVNotCanonical);

  BasicBlock *bbExit = getUniqueNonDeadEndExitBlock(loop);
  if (!bbExit)
    return emitDiagnostic(loop,
                          DiagID::ErrTapirLoopNoUniqueNonDeadEndExitBlock);

  return true;
}

PreservedAnalyses GuardLoopPass::run(Loop &loop, LoopAnalysisManager &am,
                                     LoopStandardAnalysisResults &ar,
                                     LPMUpdater &updater) {
  if (!isTapirLoop(loop) || getLoopGuardBlock(loop, /*allowDeadEndExits=*/true))
    return PreservedAnalyses::all();

  DominatorTree &dt = ar.DT;
  LoopInfo &li = ar.LI;
  MemorySSA &mssa = *ar.MSSA;

  if (!check(loop))
    exitOnError();

  InsertGuardBlock(dt, li, mssa).runOn(loop);
  return getPreservedAnalysesLoop(/*preservedMSSA=*/true);
}

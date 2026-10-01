//===- TapirLoopNormalize.cpp - Normalize tapir loops ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Normalize tapir loops. This will transform tapir loops to ensure that all of
// the following hold:
//
//   - The loop is in simplify form
//   - The loop has a canonical induction variable
//   - The termination condition occurs in the loop latch and is of the form:
//
//         %cmp = icmp eq %inc, %tc
//         br i1 %cmp, label %exit, label %header
//
//     Here, %inc is the incremented value of the canonical induction variable,
//     %tc is the loop trip count, %exit is the unique loop exit block, and
//     %header is the loop header.
//
// This pass is intended to be run just before loop-spawning. Care must be taken
// that passes such as instcombine are not run between this pass and
// loop-spawning since that may result in the loop no longer being in normalized
// form.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/TapirLoopNormalize.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/LoopUtils.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

#define DEBUG_TYPE "kit-loop-normalize"

using namespace llvm;

namespace {

class TapirLoopNormalizeImpl {
private:
  AssumptionCache &ac;
  DominatorTree &dt;
  LoopInfo &li;
  ScalarEvolution &se;

  MemorySSAUpdater mssau;

protected:
  bool check(const Loop &loop);
  Value *getOrGenerateTripCount(Loop &loop);

public:
  TapirLoopNormalizeImpl(AssumptionCache &ac, DominatorTree &dt, LoopInfo &li,
                         MemorySSA &mssa, ScalarEvolution &se)
      : ac(ac), dt(dt), li(li), se(se), mssau(&mssa) {}

  bool runOnLoop(Loop &loop);
};

} // namespace

bool TapirLoopNormalizeImpl::check(const Loop &loop) {
  // IndVarSimplifyPass should have been run before this since there is no
  // utility function to transform the induction variable of a loop so it is
  // canonical.
  if (!loop.getCanonicalInductionVariable())
    return emitDiagnostic(loop, DiagID::ErrTapirLoopIVNotCanonical);

  // There is no utility function to convert a loop to LCSSA form, so the loop
  // must already be in that form before we attempt to normalize it.
  if (!loop.isLCSSAForm(dt))
    return emitDiagnostic(loop, DiagID::ErrLoopNotLCSSAForm);

  return true;
}

Value *TapirLoopNormalizeImpl::getOrGenerateTripCount(Loop &loop) {
  assert(loop.isLoopSimplifyForm() && "Loop must be in simplify form");

  PHINode *iv = loop.getCanonicalInductionVariable();
  Type *ivTy = iv->getType();
  BasicBlock *latch = loop.getLoopLatch();
  const SCEV *exitCount = se.getExitCount(&loop, latch);
  const SCEV *tc = se.getTripCountFromExitCount(exitCount, ivTy, &loop);

  // If the trip count could not be computed, it is likely because it is not
  // loop-invariant. This should have been caught much earlier, so if it made
  // it this far, something else has gone badly wrong.
  if (isa<SCEVCouldNotCompute>(tc))
    llvm_unreachable("Could not compute tapir loop trip count");

  // If the type of the SCEV is unknown, it is an LLVM value that can be used
  // as is.
  if (auto *scUnknown = dyn_cast<SCEVUnknown>(tc))
    return scUnknown->getValue();

  // Otherwise, generate an LLVM instruction/expression out of this. Since the
  // loop is in simplify form, a preheader is guaranteed to exist.
  const DataLayout &dl = getFunction(loop)->getDataLayout();
  SCEVExpander scExpander(se, dl, "tlnorm", /*PreserveLCSSA=*/true);
  BasicBlock *ph = loop.getLoopPreheader();
  Instruction *term = ph->getTerminator();
  if (!scExpander.isSafeToExpandAt(tc, term))
    llvm_unreachable("Cannot expand trip count SCEV into tapir loop preheader");

  return scExpander.expandCodeFor(tc, ivTy, term);
}

bool TapirLoopNormalizeImpl::runOnLoop(Loop &loop) {
  if (!check(loop))
    exitOnError();

  if (isTapirLoopNormalized(loop, dt))
    return false;

  bool changed = false;
  changed |=
      simplifyLoop(&loop, &dt, &li, &se, &ac, &mssau, /*PreserveLCSSA=*/true);

  // The loop is simplified, in LCSSA form, and has a canonical induction
  // variable. The only thing stopping this from being normalized is the compare
  // instruction in the loop latch. Rather than fix up the existing instruction,
  // we just replace it with what we expect.
  BasicBlock *latch = loop.getLoopLatch();
  PHINode *iv = loop.getCanonicalInductionVariable();
  Value *ivInc = iv->getIncomingValueForBlock(latch);
  Value *tc = getOrGenerateTripCount(loop);

  ICmpInst *cmp = loop.getLatchCmpInst();
  InsertPosition insertPt = cmp->getIterator();
  ICmpInst *newCmp = new ICmpInst(insertPt, CmpInst::ICMP_EQ, ivInc, tc);

  cmp->replaceAllUsesWith(newCmp);
  cmp->eraseFromParent();

  return true;
}

PreservedAnalyses TapirLoopNormalizePass::run(Function &f,
                                              FunctionAnalysisManager &am) {
  AssumptionCache &ac = am.getResult<AssumptionAnalysis>(f);
  DominatorTree &dt = am.getResult<DominatorTreeAnalysis>(f);
  LoopInfo &li = am.getResult<LoopAnalysis>(f);
  MemorySSA &mssa = am.getResult<MemorySSAAnalysis>(f).getMSSA();
  ScalarEvolution &se = am.getResult<ScalarEvolutionAnalysis>(f);

  SmallVector<Loop *, 0> loops;
  for (Loop *loop : li.getLoopsInPreorder())
    if (isTapirLoop(*loop))
      loops.push_back(loop);

  if (loops.empty())
    return PreservedAnalyses::all();

  bool changed = false;
  TapirLoopNormalizeImpl normalizer(ac, dt, li, mssa, se);
  for (Loop *loop : loops)
    changed |= normalizer.runOnLoop(*loop);

  if (!changed)
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

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
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/PrepareTapirLoops.h"
#include "PrepareParallelLoops.h"
#include "PrepareReductionLoops.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"

#define DEBUG_TYPE "kit-prepare"

using namespace llvm;

static bool check(TapirLoopInfo &tapirLoop, bool isReduction, DominatorTree &dt,
                  LoopInfo &li) {
  if (isReduction)
    return checkReductionLoop(tapirLoop, dt, li);
  else
    return checkParallelLoop(tapirLoop, dt, li);
}

static bool prepare(TapirLoopInfo &tapirLoop, bool isReduction,
                    DominatorTree &dt, LoopInfo &li, MemorySSA &mssa,
                    ScalarEvolution &se, TaskInfo &ti) {
  if (isReduction)
    return prepareReductionLoop(tapirLoop, dt, li, mssa, se, ti);
  else
    return prepareParallelLoop(tapirLoop, dt, li, mssa, se, ti);
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
  bool isReduction = hasReductionAttr(loop);
  if (!check(tapirLoop, isReduction, dt, li))
    exitOnError();

  return prepare(tapirLoop, isReduction, dt, li, mssa, se, ti);
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
  return getLoopPassPreservedAnalyses();
}

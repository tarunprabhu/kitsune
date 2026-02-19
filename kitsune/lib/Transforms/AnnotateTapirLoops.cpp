//===- AnnotateTapirLoops.cpp - Annotate tapir loops ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass to analyze tapir loops and add appropriate annotations that will be
// used by subsequent passes. For instance, tapir loops that perform reductions
// will be annotated with a tapir.loop.reduction attribute
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/AnnotateTapirLoops.h"
#include "kitsune/Analysis/TapirLoopNestAnalysis.h"
#include "kitsune/Analysis/TapirTargetAnalysis.h"
#include "kitsune/Core/TapirLoopAttrs.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

#define DEBUG_TYPE "annotate-tapir-loops"

using namespace llvm;

/// Abstract base class for tapir loop annotator implementations.
class AnnotateTapirLoopsBase {
protected:
  LoopInfo &li;
  ScalarEvolution &se;
  TaskInfo &ti;

protected:
  AnnotateTapirLoopsBase(LoopInfo &li, ScalarEvolution &se, TaskInfo &ti)
      : li(li), se(se), ti(ti) {}

  bool isTapirLoop(Loop &loop) const { return getTaskIfTapirLoop(&loop, &ti); }

public:
  /// Annotate all tapir loops in the given function. Returns true if at least
  /// one loop was annotated, false otherwise.
  virtual bool run(Function &f, TTID tt) = 0;
};

/// Default implementation of tapir loop annotation. This simply adds a
/// tapir.loop.lower annotation to all tapir loops. This is an indication to
/// the loop spawning pass that will run later that the loops are to be lowered
/// using the tapir target specified by the tapir.loop.target annotation.
class AnnotateTapirLoopsDefault : public AnnotateTapirLoopsBase {
private:
  bool run(Loop &loop) {
    addTapirLoopLoweringEnabledAttr(loop);
    return true;
  }

public:
  AnnotateTapirLoopsDefault(LoopInfo &li, ScalarEvolution &se, TaskInfo &ti)
      : AnnotateTapirLoopsBase(li, se, ti) {}

  virtual bool run(Function &f, TTID tt) override final {
    bool changed = false;
    for (Loop *loop : li.getLoopsInPreorder())
      if (isTapirLoop(*loop))
        if (*getTapirLoopTargetAttr(*loop) == tt)
          changed |= run(*loop);
    return changed;
  }
};

/// Annotate tapir loops for use by the GPU-centric tapir targets, 'cuda' and
/// 'hip'. This only adds the perfect depth and perfect level annotations to
/// the appropriate tapir loops.
class AnnotateTapirLoopsGPU : public AnnotateTapirLoopsBase {
private:
  /// Return true if any of the ancestors of a loop are tapir loops. The given
  /// loop is not required to be a tapir loop. If the given loop is a top-level
  /// loop, return false.
  bool isAnyAncestorTapirLoop(Loop &loop) {
    Loop *parentLoop = loop.getParentLoop();
    if (!parentLoop)
      return false;
    else if (isTapirLoop(*parentLoop))
      return true;
    else
      return isAnyAncestorTapirLoop(*parentLoop);
  }

  /// Returns true if and only if the loop is a tapir loop and ANY of the
  /// following conditions hold:
  ///
  ///   - The loop is a top-level loop
  ///   - None of the ancestors of the loop are tapir loops
  ///
  bool isTopLevelTapirLoop(Loop &loop) {
    return isTapirLoop(loop) && !isAnyAncestorTapirLoop(loop);
  }

public:
  AnnotateTapirLoopsGPU(LoopInfo &li, ScalarEvolution &se, TaskInfo &ti)
      : AnnotateTapirLoopsBase(li, se, ti) {}

  bool run(Loop &root) {
    std::unique_ptr<TapirLoopNest> nest = TapirLoopNest::create(root, ti, se);
    assert(nest && "Loop must be a tapir loop");

    ArrayRef<Loop *> perfectLoops = nest->getPerfectTapirLoops();
    assert(perfectLoops.size() && "Root of tapir loop nest must be perfect");
    assert(perfectLoops[0] == &root &&
           "First perfect loop in tapir loop nest must be the root");

    // The "perfect.depth" annotation must only be set on the root of the
    // tapir loop nest. The "perfect.level" annotation must be added to all
    // loops, including the root.
    unsigned depth = nest->getMaxPerfectDepth();

    addTapirLoopLoweringEnabledAttr(root);
    addTapirLoopPerfectDepthAttr(root, depth);
    for (unsigned d = 1; d <= depth; ++d)
      addTapirLoopPerfectLevelAttr(*perfectLoops[d - 1], d);

    return true;
  }

  virtual bool run(Function &f, TTID tt) override {
    bool changed = false;
    for (Loop *loop : li.getLoopsInPreorder())
      if (isTopLevelTapirLoop(*loop))
        if (*getTapirLoopTargetAttr(*loop) == tt)
          changed |= run(*loop);
    return changed;
  }
};

/// Annotate tapir loops in the function given the primary tapir target.
/// Returns true if at least one loop in the function was annotated, false
/// otherwise.
static bool annotateTapirLoops(TTID tt, Function &f, LoopInfo &li,
                               ScalarEvolution &se, TaskInfo &ti) {
  switch (tt) {
  case TTID::Nolo:
    return false;
  case TTID::Serial:
    return AnnotateTapirLoopsDefault(li, se, ti).run(f, tt);
  case TTID::Cuda:
  case TTID::Hip:
    return AnnotateTapirLoopsGPU(li, se, ti).run(f, tt);
  case TTID::OpenCilk:
  case TTID::Pthreads:
  case TTID::Qthreads:
    return AnnotateTapirLoopsDefault(li, se, ti).run(f, tt);
  case TTID::Custom:
    // In principle, the custom tapir target is responsible for handling
    // everything related to lowering, so it is up to the tapir target to
    // handle nested parallel loops correctly. That said, it may be good to have
    // a callback, or some other hook, that could be defined by the tapir target
    // plugin and used here. For now, just use the defaults.
    return AnnotateTapirLoopsDefault(li, se, ti).run(f, tt);
  case TTID::Realm:
  case TTID::Lambda:
  case TTID::OMPTask:
  case TTID::OpenMP:
    break;
  }
  llvm_unreachable("annotateTapirLoops: TTID not handled");
}

PreservedAnalyses AnnotateTapirLoopsPass::run(Module &m,
                                              ModuleAnalysisManager &mam) {
  TapirTargetInfo &tgi = mam.getResult<TapirTargetAnalysis>(m);
  FunctionAnalysisManager &fam =
      mam.getResult<FunctionAnalysisManagerModuleProxy>(m).getManager();

  for (Function &f : m) {
    if (!f.size())
      continue;

    LoopInfo &li = fam.getResult<LoopAnalysis>(f);
    ScalarEvolution &se = fam.getResult<ScalarEvolutionAnalysis>(f);
    TaskInfo &ti = fam.getResult<TaskAnalysis>(f);

    for (TTID tt : tgi.getRequiredTTs(m))
      annotateTapirLoops(tt, f, li, se, ti);
  }

  // At best, this pass will only change the metadata on existing loops. It will
  // not add or remove any loops, or change the code itself in any other way.
  return PreservedAnalyses::all();
}

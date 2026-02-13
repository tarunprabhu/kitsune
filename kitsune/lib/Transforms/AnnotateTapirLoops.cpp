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
#include "kitsune/Core/LoopUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

#define DEBUG_TYPE "annotate-tapir-loops"

using namespace llvm;

/// Annotate tapir loops for use by the GPU-centric tapir targets, 'cuda' and
/// 'hip'. This only adds the perfect depth and perfect level annotations to
/// the appropriate tapir loops.
class AnnotateTapirLoopsGPU {
private:
  LoopInfo &li;
  ScalarEvolution &se;
  TaskInfo &ti;

private:
  bool isTapirLoop(Loop &loop) { return getTaskIfTapirLoop(&loop, &ti); }

  bool isTopLevelLoop(const Loop &loop) { return !loop.getParentLoop(); }

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
      : li(li), se(se), ti(ti) {}

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

    setTapirLoopPerfectDepthMD(root, depth);
    for (unsigned d = 1; d <= depth; ++d)
      setTapirLoopPerfectLevelMD(*perfectLoops[d - 1], d);

    return true;
  }

  bool run(Function &f) {
    bool changed = false;

    for (Loop *loop : li.getLoopsInPreorder())
      if (isTopLevelTapirLoop(*loop))
        changed |= run(*loop);

    return changed;
  }
};

/// Annotate tapir loops when the primary tapir target is 'pthreads'.
class AnnotateTapirLoopsPthreads {
public:
  AnnotateTapirLoopsPthreads(LoopInfo &, ScalarEvolution &, TaskInfo &) {}

  bool run(Function &f) {
    // We don't currently do anything special when lowering nested loops with
    // the pthreads tapir target. This might work correctly, but performance is
    // likely to be poor. This is why we create this empty stub instead of
    // just returning false from annotateTapirLoops(). However, neither claim
    // has been tested.
    return false;
  }
};

/// Annotate tapir loops in the function given the primary tapir target.
/// Returns true if at least one loop in the function was annotated, false
/// otherwise.
static bool annotateTapirLoops(TTID tt, Function &f, LoopInfo &li,
                               ScalarEvolution &se, TaskInfo &ti) {
  switch (tt) {
  case TTID::Nolo:
  case TTID::Serial:
    return false;
  case TTID::Cuda:
  case TTID::Hip:
    return AnnotateTapirLoopsGPU(li, se, ti).run(f);
  case TTID::OpenCilk:
    // Nothing to be done for the 'opencilk' tapir target. OpenCilk's runtime
    // handles this transparently using the standard work-stealing mechanisms.
    return false;
  case TTID::Pthreads:
    return AnnotateTapirLoopsPthreads(li, se, ti).run(f);
  case TTID::Custom:
    // In principle, the custom tapir target is responsible for handling
    // everything related to lowering, so it is up to the tapir target to
    // handle nested parallel loops correctly. That said, it may be good to have
    // a callback, or some other hook, that could be defined by the tapir target
    // plugin and used here.
    return false;
  case TTID::Qthreads:
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

    // At this time, we only examine the primary tapir target when determining
    // how the loops are to be annotated. At some point, we may do something
    // more sophisticated such as determining which tapir target to use
    // automatically depending on the structure and contents of the tapir loops
    // or for multi-target support.
    annotateTapirLoops(tgi.getTTID(), f, li, se, ti);
  }

  // At best, this pass will only change the metadata on existing loops. It will
  // not add or remove any loops, or change the code itself in any other way.
  return PreservedAnalyses::all();
}

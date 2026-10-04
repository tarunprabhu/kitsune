//===- PrepareParallelLoops.cpp - Transform non-reduction tapir loops -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transform tapir loops that do not perform reductions to a form that is
// suitable for parallel execution.
//
//===----------------------------------------------------------------------===//

#include "PrepareParallelLoops.h"
#include "LoopWrapping.h"
#include "PrepareTapirLoopsImpl.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/TTUtils.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"

#define DEBUG_TYPE "kit-prepare"

using namespace llvm;

static unsigned getNestDepth(Loop &loop) {
  if (std::optional<unsigned> depth = getPerfectDepthAttr(loop))
    return *depth;
  return getNestDepth(*loop.getParentLoop());
}

static bool prepareForCPU(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                          LoopInfo &li, MemorySSA &mssa, ScalarEvolution &se,
                          TaskInfo &ti) {
  return detail::PrepareTapirLoopCPU(dt, li, mssa, se, ti).run(tapirLoop);
}

static bool prepareForGPU(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                          LoopInfo &li, MemorySSA &mssa, ScalarEvolution &se,
                          TaskInfo &ti) {
  Loop &loop = *tapirLoop.getLoop();
  if (*getPerfectLevelAttr(loop) == getNestDepth(loop)) {
    return detail::PrepareTapirLoopGPU(dt, li, mssa, se, ti).run(tapirLoop);
  } else {
    addPreparedAttr(loop);
    return false;
  }
}

bool llvm::detail::prepareParallelLoop(TapirLoopInfo &tapirLoop,
                                       DominatorTree &dt, LoopInfo &li,
                                       MemorySSA &mssa, ScalarEvolution &se,
                                       TaskInfo &ti) {
  Loop &loop = *tapirLoop.getLoop();
  TTID tt = *getTargetAttr(loop);
  if (isCPUTT(tt))
    return prepareForCPU(tapirLoop, dt, li, mssa, se, ti);
  else if (isGPUTT(tt))
    return prepareForGPU(tapirLoop, dt, li, mssa, se, ti);
  else
    return false;
}

bool llvm::detail::checkParallelLoop(TapirLoopInfo &tapirLoop,
                                     DominatorTree &dt, LoopInfo &li) {
  // Currently, all checks for parallel loop are performed by the caller, since
  // they are shared between both standard parallel loops and parallel reduction
  // loops. If we ever have additional checks that are only relevant for
  // parallel loops, and not for reduction loops, those should be added here.
  return true;
}

//===- PreLowerAnnotate.cpp - Add annotations before tapir lowering -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass computes properties of tapir loops and adds them to the loop
// annotations. These will be read by passes that run later in the pipeline.
// These are intended to inform how the loop will be transformed prior to loop
// spawning and may also be used to affect how the loop will be spawned.
//
// For example, in a perfect nest of tapir loops to be compiled for the GPU,
// only the outermost loop should be handled by loop spawning - the GPU-centric
// tapir targets will correctly handle the inner loops. This pass will add the
// tapir.loop.lowering.enabled attribute to the outermost loop, but not the
// inner ones. The loop-spawning pass examines this attribute to determine which
// loops to lower.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/PreLowerAnnotate.h"
#include "kitsune/Analysis/TapirLoopNestAnalysis.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/ModuleAttrs.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Module.h"

#define DEBUG_TYPE "kit-annotate-prelower"

using namespace llvm;

PreservedAnalyses PreLowerAnnotatePass::run(Function &f,
                                            FunctionAnalysisManager &am) {
  LoopInfo &li = am.getResult<LoopAnalysis>(f);
  ScalarEvolution &se = am.getResult<ScalarEvolutionAnalysis>(f);

  /// Find the subloops that are contained within a tapir loop nest consisting
  /// of loops that are to be run on a GPU. These will be ignored.
  SmallSet<Loop *, 8> ignore;
  for (Loop *loop : li.getLoopsInPreorder())
    if (isTopLevelTapirLoopForGPU(*loop))
      for (Loop *subLoop : getAllSubLoops(*loop))
        ignore.insert(subLoop);

  // Any tapir loop that is not ignored should be annotated with the
  // tapir.loop.lowering.enabled attribute that indicates to loop-spawning that
  // the loop must be lowered.
  for (Loop *loop : li.getLoopsInPreorder())
    if (isTapirLoop(*loop) && !ignore.contains(loop))
      addLoweringEnabledAttr(*loop);

  // At best, this pass will only change the metadata on existing loops and the
  // module. It will not add or remove any loops, or change any other code.
  return PreservedAnalyses::all();
}

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
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Module.h"

#define DEBUG_TYPE "kit-annotate-prelower"

using namespace llvm;

static void addLoweringEnabledAttrs(LoopInfo &li) {
  // Tapir loop nests that will be executed on the CPU will have been
  // transformed so there is a single tapir loop over the available threads; the
  // original tapir loops will have been serialized. There will, therefore, be
  // only a single top-level tapir loop to lower. The lowering of tapir loop
  // nests on the GPU already operates only on the top-level loop.
  for (Loop *loop : getTopLevelTapirLoops(li))
    addLoweringEnabledAttr(*loop);
}

PreservedAnalyses PreLowerAnnotatePass::run(Function &f,
                                            FunctionAnalysisManager &am) {
  LoopInfo &li = am.getResult<LoopAnalysis>(f);

  addLoweringEnabledAttrs(li);

  // At most, this pass will change the metadata on existing loops and the
  // module. This will not affect any analyses.
  return PreservedAnalyses::all();
}

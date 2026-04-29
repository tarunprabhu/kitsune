//===- EarlyAnnotate.cpp - Annotator that runs early in the pipeline ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Annotator that run early in the pipeline.
//
// This is intended to run early in the pipeline to add annotations before most
// optimization passes have run. These annotations are intended to control the
// behavior of passes that run later in the pipeline.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/EarlyAnnotate.h"
#include "kitsune/Core/LoopUtils.h"
#include "llvm/Analysis/LoopInfo.h"

using namespace llvm;

static void disableUnrollingForTapirLoops(LoopInfo &li) {
  // Unconditionally disables unrolling on all tapir loops may degrade
  // performance in certain cases. In those cases, it may be more better to
  // serialize a tapir loop and allow the optimizer to unroll it. A pass that
  // performs an appropriate cost analysis and conditionally disables unrolling
  // should eventually be developed. At that time, this can be removed.
  for (Loop *loop : li.getLoopsInPreorder())
    if (isTapirLoop(*loop))
      loop->setLoopAlreadyUnrolled();
}

PreservedAnalyses EarlyAnnotatePass::run(Function &f,
                                         FunctionAnalysisManager &am) {
  LoopInfo &li = am.getResult<LoopAnalysis>(f);

  // Disable unrolling on all tapir loops. The OpenCilk compiler relies on the
  // tapir-to-target pass to handle the multiple detach instructions that result
  // from unrolling a tapir loop. However, Kitsune's tapir targets operate
  // primarily on loop nests and require these to have a single detach
  // instruction. Disabling unrolling is the only way to ensure that, especially
  // at higher optimization levels, we do not end up with a tapir loop nest
  // that Kitsune's tapir targets are unable to process.
  disableUnrollingForTapirLoops(li);

  // This only adds metadata that should not invalidate any analyses.
  return PreservedAnalyses::all();
}

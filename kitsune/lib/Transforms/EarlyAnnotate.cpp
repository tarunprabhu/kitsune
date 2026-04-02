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
// optimization passes have run. These annotations are typically
// Kitsune-specific instruction attributes, but they need not be just those.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/EarlyAnnotate.h"
#include "kitsune/Core/InstAttrs.h"
#include "kitsune/Core/LoopAttrs.h"
#include "llvm/Analysis/LoopInfo.h"

using namespace llvm;

PreservedAnalyses EarlyAnnotatePass::run(Function &f,
                                         FunctionAnalysisManager &am) {
  LoopInfo &li = am.getResult<LoopAnalysis>(f);

  // Add the source loop attribute to instructions. We only do so for
  // instructions contained within tapir loops.
  for (Loop *loop : li.getLoopsInPreorder())
    if (hasTargetAttr(*loop))
      for (BasicBlock *bb : loop->getBlocks())
        if (li.getLoopFor(bb) == loop)
          for (Instruction &inst : *bb)
            if (!hasSourceLoopAttr(inst))
              addSourceLoopAttr(inst, loop->getLoopID());

  // This only adds metadata that should not invalidate any analyses.
  return PreservedAnalyses::all();
}

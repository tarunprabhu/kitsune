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
// This is intended to run early in the pipeline, typically after mem2reg (or
// the equivalent).
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/EarlyAnnotate.h"
#include "kitsune/Core/LoopUtils.h"
#include "llvm/Analysis/LoopInfo.h"

using namespace llvm;

PreservedAnalyses EarlyAnnotatePass::run(Function &f,
                                         FunctionAnalysisManager &am) {
  LoopInfo &li = am.getResult<LoopAnalysis>(f);

  for (Loop *loop : li.getLoopsInPreorder()) {
    if (isTapirLoop(*loop)) {
      addMandatoryLLVMLoopAttrs(*loop);
    }
  }

  // At this time, the added annotations do not invalidate other analyses.
  // However, if we ever add any that do such as !tbaa (which affects alias
  // analysis), this must be changed.
  return PreservedAnalyses::all();
}

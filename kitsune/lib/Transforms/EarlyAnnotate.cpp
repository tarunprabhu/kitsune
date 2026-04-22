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
  // This pass was initially added as a requirement for the kit-delicm pass.
  // But the implementation of kit-delicm no longer requires this. As a result,
  // this does nothing. We leave it around in case we have some use for it in
  // the future.

  // This only adds metadata that should not invalidate any analyses.
  return PreservedAnalyses::all();
}

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
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Intrinsics.h"

#define DEBUG_TYPE "kit-annotate-early"

using namespace llvm;

static bool isReductionLoop(const Loop &loop) {
  for (const BasicBlock *bb : loop.getBlocks())
    for (const Instruction &inst : *bb)
      if (const auto *call = dyn_cast<CallBase>(&inst))
        if (call->getIntrinsicID() == Intrinsic::kit_reduce_0)
          return true;
  return false;
}

PreservedAnalyses EarlyAnnotatePass::run(Function &f,
                                         FunctionAnalysisManager &am) {
  LoopInfo &li = am.getResult<LoopAnalysis>(f);

  for (Loop *loop : li.getLoopsInPreorder()) {
    if (isTapirLoop(*loop)) {
      LLVM_DEBUG(dbgs() << "EarlyAnnotate: Annotating tapir loop '"
                        << getName(*loop) << "'\n");
      addMandatoryLLVMLoopAttrs(*loop);
      if (isReductionLoop(*loop))
        addReductionAttr(*loop);
    }
  }

  // At this time, the added annotations do not invalidate other analyses.
  // However, if we ever add any that do such as !tbaa (which affects alias
  // analysis), this must be changed.
  return PreservedAnalyses::all();
}

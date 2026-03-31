//=- EarlyAnnotate.h - Annotator that runs early in the pipeline --*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Annotator that run early in the pipeline
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_EARLY_ANNOTATE_H
#define KITSUNE_TRANSFORMS_EARLY_ANNOTATE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// Annotator pass that runs early in the pipeline.
class EarlyAnnotatePass : public PassInfoMixin<EarlyAnnotatePass> {
public:
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &am);
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_EARLY_ANNOTATE_H

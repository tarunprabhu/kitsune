//=- AnnotateEarly.h - Annotator that runs early in the pipeline --*- C++ -*-=//
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

#ifndef KITSUNE_TRANSFORMS_ANNOTATE_EARLY_H
#define KITSUNE_TRANSFORMS_ANNOTATE_EARLY_H

#include "kitsune/Core/FuncAttrs.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// \ingroup kitsune
/// Annotator pass that runs early in the pipeline.
class AnnotateEarlyPass : public PassInfoMixin<AnnotateEarlyPass> {
public:
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &am);

public:
  static constexpr FuncAttrKind hasRunAttr = FuncAttrKind::AnnotateEarlyPass;
};

static_assert(check_pass_requirable<AnnotateEarlyPass>());

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_ANNOTATE_EARLY_H

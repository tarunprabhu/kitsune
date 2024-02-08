//==- PreLowerAnnotate.h - Add annotations before tapir lowering -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Add Kitsune-specific annotations just before tapir loop lowering.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_PRE_LOWER_ANNOTATE_H
#define KITSUNE_TRANSFORMS_PRE_LOWER_ANNOTATE_H

#include "kitsune/Core/FuncAttrs.h"
#include "kitsune/Passes/PassUtils.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// \ingroup kitsune
/// Add Kitsune-specific annotations that will be used by passes that run later
/// in the pipeline. This pass is intended to run just before tapir loop
/// lowering.
class PreLowerAnnotatePass : public PassInfoMixin<PreLowerAnnotatePass> {
public:
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &am);

  static constexpr auto hasRunAttr = FuncAttrKind::PreLowerAnnotatePass;
};

static_assert(check_pass_requirable<PreLowerAnnotatePass>());

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_PRE_LOWER_ANNOTATE_H

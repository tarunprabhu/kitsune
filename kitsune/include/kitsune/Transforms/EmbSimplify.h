//===- EmbSimplify.h - Simplify functions in embedded modules ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Run a sequence of function simplification passes on functions in embedded
// modules.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_EMB_SIMPLIFY_H
#define KITSUNE_TRANSFORMS_EMB_SIMPLIFY_H

#include "kitsune/Passes/EmbPass.h"

namespace llvm {

/// \ingroup
/// Run a sequence of function simplification passes on functions in an embedded
/// module.
class EmbSimplifyPass : public EmbFunctionPass<EmbSimplifyPass> {
public:
  bool run(TTID tt, Function &devF, FunctionAnalysisManager &devAM,
           Module &hostM, ModuleAnalysisManager &hostAM);

  using EmbFunctionPass<EmbSimplifyPass>::run;
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_EMB_SIMPLIFY_H

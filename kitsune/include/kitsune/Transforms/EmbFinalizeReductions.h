//=- EmbFinalizeReductions.h - Finalize reductions in device code -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Finalize reductions in embedded modules.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_EMB_FINALIZE_REDUCTIONS_H
#define KITSUNE_TRANSFORMS_EMB_FINALIZE_REDUCTIONS_H

#include "kitsune/Passes/EmbPass.h"

namespace llvm {

/// \ingroup kitsune
/// Finalize reductions in an embedded module.
class EmbFinalizeReductionsPass
    : public EmbFunctionPass<EmbFinalizeReductionsPass> {
public:
  bool run(TTID tt, Function &devF, FunctionAnalysisManager &devAM,
           Module &hostM, ModuleAnalysisManager &hostAM);

  using EmbFunctionPass<EmbFinalizeReductionsPass>::run;
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_EMB_FINALIZE_REDUCTIONS_H

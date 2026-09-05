//===- EmbFinalizeReductions.h - Finalize reduction kernels ----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Finalize GPU kernels that perform a reduction. These kernels will have been
// obtained from tapir loops that contain reductions.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_EMB_FINALIZE_REDUCTIONS_H
#define KITSUNE_TRANSFORMS_EMB_FINALIZE_REDUCTIONS_H

#include "kitsune/Passes/EmbModulePass.h"

namespace llvm {

/// \ingroup kitsune
/// Lower Kitsune-specific intrinsics in an embedded module.
class EmbFinalizeReductionsPass
    : public EmbModulePass<EmbFinalizeReductionsPass> {
public:
  bool run(TTID tt, Module &devM, ModuleAnalysisManager &devAM, Module &hostM,
           ModuleAnalysisManager &hostAM);

  using EmbModulePass<EmbFinalizeReductionsPass>::run;
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_EMB_FINALIZE_REDUCTIONS_H

//===- EmbPrepare.h - Prepare embedded modules for codegen -----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Prepare embedded modules for code generation.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_EMB_PREPARE_H
#define KITSUNE_TRANSFORMS_EMB_PREPARE_H

#include "kitsune/Transforms/EmbModulePass.h"

namespace llvm {

/// \ingroup kitsune
/// Prepare the embedded bitcode for code generation.
class EmbPreparePass : public EmbModulePass<EmbPreparePass> {
public:
  bool run(TTID tt, Module &devM, Module &hostM, ModuleAnalysisManager &hostAM);

  using EmbModulePass<EmbPreparePass>::run;
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_EMB_PREPARE_H

//===- EmbLowerKitsuneIntrinsics.h - Lower Kitsune's intrinsics -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune-specific intrinsics in embedded modules.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CODEGEN_EMB_LOWER_KITSUNE_INTRINSICS_H
#define KITSUNE_CODEGEN_EMB_LOWER_KITSUNE_INTRINSICS_H

#include "kitsune/Passes/EmbModulePass.h"

namespace llvm {

class ModulePass;

/// \ingroup kitsune
/// Lower Kitsune-specific intrinsics in embedded modules.
class EmbLowerKitsuneIntrinsicsPass
    : public EmbModulePass<EmbLowerKitsuneIntrinsicsPass> {
public:
  bool run(TTID tt, Module &devM, Module &hostM,
           ModuleAnalysisManager &hostMAM);

  using EmbModulePass<EmbLowerKitsuneIntrinsicsPass>::run;
};

/// \ingroup kitsune
ModulePass *createEmbLowerKitsuneIntrinsicsLegacyPass();

} // namespace llvm

#endif // KITSUNE_CODEGEN_EMB_LOWER_KITSUNE_INTRINSICS_H

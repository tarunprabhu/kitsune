//=- EmbLowerKitIntrinsics.h - Lower Kitsune-specific intrinsics -*- C++ -*--=//
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

#ifndef KITSUNE_CODEGEN_EMB_LOWER_KIT_INTRINSICS_H
#define KITSUNE_CODEGEN_EMB_LOWER_KIT_INTRINSICS_H

#include "kitsune/Passes/EmbModulePass.h"

namespace llvm {

class ModulePass;

/// \ingroup kitsune
/// Lower Kitsune's intrinsics in embedded modules.
class EmbLowerKitIntrinsicsPass
    : public EmbModulePass<EmbLowerKitIntrinsicsPass> {
public:
  bool run(TTID tt, Module &devM, Module &hostM, ModuleAnalysisManager &hostAM);

  using EmbModulePass<EmbLowerKitIntrinsicsPass>::run;
};

/// \ingroup kitsune
/// Create a legacy pass to lower Kitsune's intrinsics in an embedded module.
ModulePass *createEmbLowerKitIntrinsicsLegacyPass();

} // namespace llvm

#endif // KITSUNE_CODEGEN_EMB_LOWER_KIT_INTRINSICS_H

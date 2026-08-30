//===- EmbLowerIntrinsicsEarly.h - Lower Kitsune intrinsics ----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune-specific intrinsics in an embedded module. Although most
// Kitsune-specific intrinsics are lowered by the backend, this is specifically
// intended to run as part of Kitsune's lowering pipeline in the middle-end.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_EMB_LOWER_INTRINSICS_EARLY_H
#define KITSUNE_TRANSFORMS_EMB_LOWER_INTRINSICS_EARLY_H

#include "kitsune/Passes/EmbModulePass.h"

namespace llvm {

/// \ingroup kitsune
/// Lower Kitsune-specific intrinsics in an embedded module. This is intended
/// to be run as part of Kitsune's lowering pipeline in the middle-end. It is
/// only for intrinsics that, for whatever reason, must be lowered early unlike
/// the standard lowering of Kitsune's intrinsics which is carried out in the
/// backend.
class EmbLowerIntrinsicsEarlyPass
    : public EmbModulePass<EmbLowerIntrinsicsEarlyPass> {
public:
  bool run(TTID tt, Module &devM, Module &hostM, ModuleAnalysisManager &hostAM);

  using EmbModulePass<EmbLowerIntrinsicsEarlyPass>::run;
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_EMB_LOWER_INTRINSICS_EARLY_H

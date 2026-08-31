//===- EmbLowerIntrinsics.h - Lower Kitsune intrinsics ---------*- C++ -*--===//
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

#ifndef KITSUNE_TRANSFORMS_EMB_LOWER_INTRINSICS_H
#define KITSUNE_TRANSFORMS_EMB_LOWER_INTRINSICS_H

#include "kitsune/Passes/EmbModulePass.h"

namespace llvm {

/// \ingroup kitsune
/// Lower Kitsune-specific intrinsics in an embedded module.
class EmbLowerIntrinsicsPass : public EmbModulePass<EmbLowerIntrinsicsPass> {
protected:
  const TTOptions &tto;

public:
  EmbLowerIntrinsicsPass(const TTOptions &tto) : tto(tto) {}

  bool run(TTID tt, Module &devM, Module &hostM, ModuleAnalysisManager &hostAM);

  using EmbModulePass<EmbLowerIntrinsicsPass>::run;
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_EMB_LOWER_INTRINSICS_H

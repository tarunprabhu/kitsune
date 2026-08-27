//- EmbLowerKitWarpIntrinsics.h - Lower Kitsune's warp intrinsics -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune's warp intrinsics.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_EMB_LOWER_KIT_WARP_INTRINSICS_H
#define KITSUNE_TRANSFORMS_EMB_LOWER_KIT_WARP_INTRINSICS_H

#include "kitsune/Passes/EmbModulePass.h"

namespace llvm {

class TTOptions;

/// \ingroup kitsune
/// Lower Kitsune's warp intrinsics.
class EmbLowerKitWarpIntrinsicsPass
    : public EmbModulePass<EmbLowerKitWarpIntrinsicsPass> {
protected:
  const TTOptions &tto;

public:
  EmbLowerKitWarpIntrinsicsPass(const TTOptions &tto) : tto(tto) {}

  bool run(TTID tt, Module &devM, Module &hostM, ModuleAnalysisManager &hostAM);

  using EmbModulePass<EmbLowerKitWarpIntrinsicsPass>::run;
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_EMB_LOWER_KIT_WARP_INTRINSICS_H

//===- EmbLowerKitIntrinsicsLibDevice.h - Lower intrinsics -----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower those Kitsune-specific intrinsics in an embedded module that must be
// lowered to functions provided by a libdevice bitcode file.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_EMB_LOWER_KIT_INTRINSICS_LIBDEVICE_H
#define KITSUNE_TRANSFORMS_EMB_LOWER_KIT_INTRINSICS_LIBDEVICE_H

#include "kitsune/Passes/EmbModulePass.h"

namespace llvm {

/// \ingroup kitsune
/// Lower those Kitsune-specific intrinsics in an embedded module that must be
/// lowered to functions provided by a libdevice bitcode file.
class EmbLowerKitIntrinsicsLibDevicePass
    : public EmbModulePass<EmbLowerKitIntrinsicsLibDevicePass> {
public:
  bool run(TTID tt, Module &km, Module &hostM, ModuleAnalysisManager &hostMAM);

  using EmbModulePass<EmbLowerKitIntrinsicsLibDevicePass>::run;
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_EMB_LOWER_KIT_INTRINSICS_LIB_DEVICE_H

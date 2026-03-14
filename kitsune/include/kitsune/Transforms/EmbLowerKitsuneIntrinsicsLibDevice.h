//==- EmbLowerKitsuneIntrinsicsLibDevice.h - Lower intrinsics ---*- C++ -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune-specific intrinsics in an embedded module that must use
// functions provided by a libdevice bitcode file.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_EMB_LOWER_KITSUNE_INTRINSICS_LIBDEVICE_H
#define KITSUNE_TRANSFORMS_EMB_LOWER_KITSUNE_INTRINSICS_LIBDEVICE_H

#include "kitsune/Passes/EmbModulePass.h"

namespace llvm {

/// \ingroup kitsune
/// Lower Kitsune-specific intrinsics in an embedded module that must use
/// functions provided by a libdevice bitcode file.
class EmbLowerKitsuneIntrinsicsLibDevicePass
    : public EmbModulePass<EmbLowerKitsuneIntrinsicsLibDevicePass> {
public:
  bool run(TTID tt, Module &km, Module &hostM, ModuleAnalysisManager &hostMAM);

  using EmbModulePass<EmbLowerKitsuneIntrinsicsLibDevicePass>::run;
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_EMB_LOWER_KITSUNE_INTRINSICS_LIB_DEVICE_H

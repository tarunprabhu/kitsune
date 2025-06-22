//=- LinkDeviceBitcode.h - Link device and embedded bc moodules ---*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Link the device bitcode module(s) into the embedded bitcode modules.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_LINK_DEVICE_BITCODE_H
#define KITSUNE_TRANSFORMS_LINK_DEVICE_BITCODE_H

#include "kitsune/Transforms/EmbBCPass.h"

namespace llvm {

/// Link the appropriate device bitcode files into the embedded bitcode.
class LinkDeviceBitcodePass : public EmbBCPass<LinkDeviceBitcodePass> {
public:
  bool run(TTID tt, Module &km, Module &hostM, ModuleAnalysisManager &hostMAM);

  using EmbBCPass<LinkDeviceBitcodePass>::run;
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_LINK_DEVICE_BITCODE_H

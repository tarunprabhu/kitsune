//=- LinkDeviceBitcode.h - Link device bitcode into embedded bitcode --------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Link the device bitcode file(s) into the embedded bitcode modules.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_KITSUNE_LINK_DEVICE_BITCODE_H
#define LLVM_TRANSFORMS_KITSUNE_LINK_DEVICE_BITCODE_H

#include "llvm/Transforms/Kitsune/EmbBCPass.h"

namespace llvm {

/// Link the appropriate device bitcode files into the embedded bitcode.
class LinkDeviceBitcodePass : public EmbBCPass<LinkDeviceBitcodePass> {
public:
  bool run(TapirTargetID tt, Module &km, Module &hostM,
           ModuleAnalysisManager &hostMAM);

  using EmbBCPass<LinkDeviceBitcodePass>::run;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_KITSUNE_LINK_DEVICE_BITCODE_H

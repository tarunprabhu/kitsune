//- EmbLinkLibDeviceBitcode.h - Link libdevice & embedded bitcode -*- C++ -*--//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Link the libdevice bitcode module(s) into the embedded bitcode modules.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_EMB_LINK_LIBDEVICE_BITCODE_H
#define KITSUNE_TRANSFORMS_EMB_LINK_LIBDEVICE_BITCODE_H

#include "kitsune/Passes/EmbModulePass.h"

namespace llvm {

class TTOptions;

/// \ingroup kitsune
/// Link the appropriate device bitcode modules into the embedded bitcode.
class EmbLinkLibDeviceBitcodePass
    : public EmbModulePass<EmbLinkLibDeviceBitcodePass> {
protected:
  const TTOptions &tto;

public:
  EmbLinkLibDeviceBitcodePass(const TTOptions &tto) : tto(tto) {}

  bool run(TTID tt, Module &km, Module &hostM, ModuleAnalysisManager &hostMAM);

  using EmbModulePass<EmbLinkLibDeviceBitcodePass>::run;
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_EMB_LINK_LIBDEVICE_BITCODE_H

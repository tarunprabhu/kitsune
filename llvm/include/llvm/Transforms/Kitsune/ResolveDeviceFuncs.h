//=- ResolveDeviceFuncsInEmbBC.h - Resolve device functions ------*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The embedded bitcode may contain calls to library functions for which
// device-specific implementations exist. This resolves the calls to such
// functions in the embedded bitcode to use the device-specific implementations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_KITSUNE_RESOLVE_DEVICE_FUNCS_H
#define LLVM_TRANSFORMS_KITSUNE_RESOLVE_DEVICE_FUNCS_H

#include "llvm/Transforms/Kitsune/EmbBCPass.h"

namespace llvm {

/// Resolve the calls to functions in the embedded bitcode that have
/// device-specific implementations in one or more vendor-provided bitcode files
/// for the device (usually a GPU). This will look for calls to functions that
/// have device equivalents, add declarations for the called device equivalents,
/// then replace the calls with these new declarations. In some cases, bitcode
/// files are provided by vendors containing the definitions of these functions.
/// Those bitcode files are linked into the embedded bitcode module in a
/// separate pass.
class ResolveDeviceFuncsPass : public EmbBCPass<ResolveDeviceFuncsPass> {
public:
  bool run(TapirTargetID tt, Module &km, Module &hostM,
           ModuleAnalysisManager &hostMAM);

  using EmbBCPass<ResolveDeviceFuncsPass>::run;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_KITSUNE_RESOLVE_DEVICE_FUNCS_H

//- KitsuneLoweringUtils.h - Utilities for Kitsune's tapir targets -*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for Kitsune's tapir targets. It may be possible to move this to one
// of Kitsune's core/support libraries, but since the tapir targets live in
// llvm/lib, this is retained here.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TAPIR_KITSUNE_LOWERING_UTILS_H
#define LLVM_TRANSFORMS_TAPIR_KITSUNE_LOWERING_UTILS_H

#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/IRBuilder.h"

namespace llvm {

class Loop;
class Module;
class ReachableGlobals;

/// Construct the name to be used for the outlined function consisting of the
/// body of the tapir loop.
std::string getNameForTapirLoop(const Loop &loop, StringRef prefix = "",
                                unsigned suffix = 0);

/// Construct the name for a device module.
std::string getNameForDeviceModule(const Module &hostM, StringRef prefix = "");

/// Generate calls to copy non-constant globals, which are used in the outlined
/// tapir loop, from device to host after launching the kernel.
void copyNonConstGlobalsDToH(const ReachableGlobals &globals, TTID tt,
                             Module &m, IRBuilder<> &builder);

/// Generate calls to copy non-constant globals, which are used in the outlined
/// tapir loop, from host to device before launching the kernel.
void copyNonConstGlobalsHToD(const ReachableGlobals &globals, TTID tt,
                             Module &m, IRBuilder<> &builder);

} // namespace llvm

#endif // LLVM_TRANSFORMS_TAPIR_KITSUNE_LOWERING_UTILS_H

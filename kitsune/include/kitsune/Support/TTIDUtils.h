//===- TTIDUtils.h - Utilities to deal with tapir target ids ---*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for TTID's
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_SUPPORT_TTID_UTILS_H
#define KITSUNE_SUPPORT_TTID_UTILS_H

#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/ArrayRef.h"

namespace llvm {

/// \addtogroup kitsune
/// @{

/// Get the tapir targets that generate embedded bitcode.
ArrayRef<TTID> ttsGenEmbBC();

/// Check if the given tapir target generates embedded bitcode.
bool doesTTGenEmbBC(TTID tt);
bool generatesEmbBC(TTID tt);

/// Does the tapir target generate code that will run on a GPU.
bool isGPUTT(TTID tt);

/// @}

} // namespace llvm

#endif // KITSUNE_SUPPORT_TTID_UTILS_H

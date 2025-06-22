//===- TTUtils.h - Utilities to deal with tapir target ids -----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to get "properties" of tapir targets.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_SUPPORT_TTUTILS_H
#define KITSUNE_SUPPORT_TTUTILS_H

#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/ArrayRef.h"

namespace llvm {

/// Get all known tapir targets.
/// FIXME: The definition of this function must be updated when a new tapir
/// target is added.
ArrayRef<TTID> ttsAll();

/// Get the tapir targets that generate embedded bitcode.
ArrayRef<TTID> ttsGenEmbBC();

/// Check if the given tapir target generates embedded bitcode.
bool doesTTGenEmbBC(TTID tt);

} // namespace llvm

#endif // KITSUNE_SUPPORT_TTUTILS_H

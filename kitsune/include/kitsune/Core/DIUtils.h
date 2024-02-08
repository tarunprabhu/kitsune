//===- DIUtils.h - Utilities for DebugInfo ---------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to work with debug information. Some of these are wrappers around
// LLVM utilities that are otherwise slightly awkward to work with, while others
// are Kitsune-specific customizations that work better for us.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_DI_UTILS
#define KITSUNE_CORE_DI_UTILS

#include "llvm/ADT/StringRef.h"

namespace llvm {

class DebugLoc;

/// \addtogroup kitsune
/// @{

/// Convert the given debug location, \p dbgLoc to a string. If \p dbgLoc is not
/// valid, or if the file name could not be determined,return an empty string.
/// Otherwise, compute the base location which will be of the form
/// "<file>:<line>:<col>". If the column number is available, "<file>:<line>"
/// otherwise. If \p inlinedAt is `false`, just return the base location.
/// Otherwise, call this function with the inlined location. If this returns a
/// non-empty string, <inlined>, append "@[<inlined>]" to the base string and
/// return the result.
std::string toString(const DebugLoc &dbgLoc, bool inlinedAt = false);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_DI_UTILS

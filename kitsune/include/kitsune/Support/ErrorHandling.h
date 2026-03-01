//===- ErrorHandling.h - Utilities for abnormal exits ----------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to deal with abnormal exists. These are slight variations on those
// provided by LLVM that are better suited for Kitsune.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_SUPPORT_ERROR_HANDLING_H
#define KITSUNE_SUPPORT_ERROR_HANDLING_H

#include "llvm/Support/Error.h"

namespace llvm {

/// \addtogroup kitsune
/// @{

/// Exit with a system-dependent error code. This will exit silently, so it is
/// recommended to emit a diagnostic before calling this function.
[[noreturn]]
void exitOnError();

/// Convert the given error to a string, emit it to stderr, then exit with a
/// system-dependent error code.
[[noreturn]]
void exitOnError(Error e);

/// @}

} // namespace llvm

#endif // KITSUNE_SUPPORT_ERROR_HANDLING_H

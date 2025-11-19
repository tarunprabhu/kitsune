//===- ErrorUtils.h - Utilities for LLVM's error objects -------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM's error objects
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_SUPPORT_ERROR_UTILS_H
#define KITSUNE_SUPPORT_ERROR_UTILS_H

namespace llvm {

class Error;

/// Ignore all the errors in the given error. This simply ensures that the
/// errors are handled. This should only be used in cases where an error could
/// occur, but we know that the same error will be handled correctly elsewhere.
void ignoreAllErrors(Error &&err);

} // namespace llvm

#endif // KITSUNE_SUPPORT_ERROR_UTILS_H

//==- TapirCommandLineUtils.h - Parse tapir-specific cl options -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Helper functions to parse tapir-specific command line options
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TAPIR_COMMAND_LINE_UTILS_H
#define LLVM_TAPIR_COMMAND_LINE_UTILS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Transforms/Tapir/TapirTargetIDs.h"

namespace llvm {

/// Parse the tapir target from a string. If the string is not a valid tapir
/// target, return std::nullopt.
std::optional<TapirTargetID> parseTapirTarget(StringRef s);

/// Parse an optional boolean from a string. The value returned is according to
/// the table below.
///
///     off  false
///     on   true
///     any  std::nullopt
///
/// If the value was not any of the values above, an invalid argument error is
/// returned.
llvm::ErrorOr<std::optional<bool>> parseOptionalBool(StringRef s);

} // namespace llvm

#endif // LLVM_TAPIR_COMMAND_LINE_UTILS_H

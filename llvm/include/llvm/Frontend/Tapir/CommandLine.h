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

#ifndef LLVM_TAPIR_COMMAND_LINE_H
#define LLVM_TAPIR_COMMAND_LINE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/Tapir/Tapir.h"
#include "llvm/Support/ErrorOr.h"

namespace llvm {

/// Parse the tapir target from a string. If the string is not a valid tapir
/// target, return std::nullopt.
ErrorOr<TapirTargetID> parseTapirTarget(StringRef s);

/// Parse a @ref MaybeBool enum from a string. If the string is not a valid
/// string for this enum, an invalid argument error is returned.
ErrorOr<MaybeBool> parseMaybeBool(StringRef s);

} // namespace llvm

#endif // LLVM_TAPIR_COMMAND_LINE_H

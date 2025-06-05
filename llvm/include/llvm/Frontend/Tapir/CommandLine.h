//==- CommandLine.h - Utilities for tapir-specific cl options ---*- C++ -*-===//
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
#include "llvm/Support/CommandLine.h"

#include <optional>

namespace llvm {

/// Parse the tapir target from a string. If the string is not a valid tapir
/// target, return std::nullopt.
ErrorOr<TapirTargetID> parseTapirTarget(StringRef s);

/// Parse a @ref MaybeBool enum from a string. If the string is not a valid
/// string for this enum, an invalid argument error is returned.
ErrorOr<MaybeBool> parseMaybeBool(StringRef s);

namespace cl {

/// Parser for command line options that will create an optional TapirTargetID.
struct TapirTargetIDParser : public cl::parser<std::optional<TapirTargetID>> {
  TapirTargetIDParser(
      cl::opt<std::optional<TapirTargetID>, false, TapirTargetIDParser> &opt)
      : parser(opt) {}
  bool parse(cl::Option &opt, StringRef name, StringRef val,
             std::optional<TapirTargetID> &result) {
    result = std::nullopt;
    if (ErrorOr<TapirTargetID> tt = parseTapirTarget(val)) {
      result = *tt;
    } else {
      opt.error("invalid value '" + val + "' in '" + name + "'");
    }
    return !result.has_value();
  }
};

} // namespace cl

} // namespace llvm

#endif // LLVM_TAPIR_COMMAND_LINE_H

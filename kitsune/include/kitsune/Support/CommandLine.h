//=- CommandLine.h - Kitsune-specific shared command line options -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Kitsune-specific command line options and utilities that are shared across
// tools.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_SUPPORT_COMMAND_LINE_H
#define KITSUNE_SUPPORT_COMMAND_LINE_H

#include "kitsune/Core/Tapir.h"
#include "llvm/Support/CommandLine.h"

#include <optional>

namespace llvm {

/// Get the category for the shared kitsune-specific options. These will contain
/// options that may be used by more than one tool.
cl::OptionCategory &getKitClOptCategory();

/// Parse the value of the --tapir command line option. If the option was not
/// provided, a default will be returned. If the option was provided and the
/// value does not correspond to a valid tapir target, an error will be raised.
std::optional<TTID> getClOptTapir(std::optional<TTID> defawlt = std::nullopt);

} // namespace llvm

#endif // KITSUNE_SUPPORT_COMMAND_LINE_H

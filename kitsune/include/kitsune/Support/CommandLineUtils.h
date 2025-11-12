//=- CommandLineUtils.h - Utilities for command line options ------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM's command line options.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_SUPPORT_COMMAND_LINE_UTILS_H
#define KITSUNE_SUPPORT_COMMAND_LINE_UTILS_H

#include "llvm/ADT/StringRef.h"

namespace llvm {

// NOTE: Most options that control Kitsune's behavior are defined in
// TTOptions.cpp. They are private to the file and we intend to keep it that
// way. These options are in the "Kitsune" option category.
//
// In the Kitsune tools, we hide all categories except those of the tool's
// command line options. This keeps the help screen of the tools clean. However,
// some options from the main "Kitsune" category are needed by the tools,
// typically the --tapir option. To enable this, we declare some helper
// functions to override the visibility and description of these options. Since
// we don't expect the spelling of these options to change, we require the
// string representing the option to be passed to these utilities.

/// Make the option with the given spelling visible.
void clSetOptionVisible(StringRef spelling);

/// Override the description of the option with the given spelling.
void clSetOptionDescription(StringRef spelling, StringRef descr);

} // namespace llvm

#endif // KITSUNE_SUPPORT_COMMAND_LINE_UTILS_H

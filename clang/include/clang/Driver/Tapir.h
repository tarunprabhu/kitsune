//===--- Tapir.h - Parse kitsune-specific command line options --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Helper functions to parse Kitsune-specific command line options
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_TAPIR_H
#define LLVM_CLANG_DRIVER_TAPIR_H

#include "llvm/Frontend/Tapir/Tapir.h"

namespace llvm {
namespace opt {
class Arg;
class ArgList;
} // namespace opt
} // namespace llvm

namespace clang {

/// Parse the --tapir flag if it is present and return the tapir target ID. This
/// should only be called when the value of the tapir --tapir flag is valid.
std::optional<llvm::TapirTargetID>
parseTapirTargetIfValid(const llvm::opt::ArgList &Args);

/// Parse the -ftapir flag if it is present and get the name of the config file
/// of the Tapir target that was specified. If the argument of the -ftapir flag
/// is invalid, this will return std::nullopt.
std::optional<llvm::StringRef>
getTapirTargetConfigFileName(const llvm::opt::ArgList &Args);

} // namespace clang

#endif

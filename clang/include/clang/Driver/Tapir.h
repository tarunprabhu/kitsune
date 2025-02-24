//===--- Tapir.h - C Language Family Language Options -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Defines helper functions for processing flags related to Tapir.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_TAPIR_H
#define LLVM_CLANG_DRIVER_TAPIR_H

#include "llvm/Transforms/Tapir/TapirTargetIDs.h"

namespace llvm {
namespace opt {
class Arg;
class ArgList;
} // namespace opt
} // namespace llvm

namespace clang {

/// Parse the -ftapir flag from the given argument. If the value is not a known
/// tapir target name, return std::nullopt.
std::optional<llvm::TapirTargetID> parseTapirTarget(const llvm::opt::Arg &A);

/// Parse the -ftapir flag if it is present in the arguments. If the flag is
/// not present, or if it is not a known tapir target name, return std::nullopt.
std::optional<llvm::TapirTargetID>
parseTapirTarget(const llvm::opt::ArgList &Args);

/// Parse the -ftapir-cuda-arch flag from the given argument. If the value is
/// not a known cuda architecture name, return an empty string.
llvm::StringRef parseTapirCudaArch(const llvm::opt::Arg &A);

/// Parse the -ftapir-cuda-arch flag if it is present in the arguments. If the
/// flag is not present, return std::nullopt. If the flag is present and its
/// value is not valid, return "".
std::optional<std::string> parseTapirCudaArch(const llvm::opt::ArgList &Args);

/// Parse the -ftapir-hip-arch flag from the given argument. If the value is
/// not a known hip architecture name, return an empty string.
llvm::StringRef parseTapirHipArch(const llvm::opt::Arg &A);

/// Parse the -ftapir-hip-arch flag if it is present in the arguments. If the
/// flag is not present, return std::nullopt. If the flag is present and its
/// value is not valid, return "".
std::optional<std::string> parseTapirHipArch(const llvm::opt::ArgList &Args);

/// Parse the -ftapir flag if it is present and get the name of the config file
/// of the Tapir target that was specified. If the argument of the -ftapir flag
/// is invalid, this will return std::nullopt.
std::optional<llvm::StringRef>
getTargetConfigFileName(const llvm::opt::ArgList &Args);

} // namespace clang

#endif

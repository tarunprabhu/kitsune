//===- TapirUtils.h - String utility methods for Tapir ---------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility functions to convert enums and types to (usually) a user-friendly
// string.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TAPIR_STRING_UTILS_H
#define LLVM_TAPIR_STRING_UTILS_H

#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Tapir/TapirTargetIDs.h"

namespace llvm {

// Serialization functions to help in debugging/verbose mode. Some of these
// are the inverse of a parse* function which parses a command line option.
raw_ostream &operator<<(raw_ostream &os, const bool &);
raw_ostream &operator<<(raw_ostream &os, const OptimizationLevel &);
raw_ostream &operator<<(raw_ostream &os, const FPOpFusion::FPOpFusionMode &);
raw_ostream &operator<<(raw_ostream &os, const TapirTargetID &);
raw_ostream &operator<<(raw_ostream &os, const TapirSpawnStrategy &);
raw_ostream &operator<<(raw_ostream &os, const std::optional<bool> &);
raw_ostream &operator<<(raw_ostream &os, const std::optional<TapirTargetID> &);

namespace tapir {

// Helper function for @ref concat. This is the base case. We don't bother with
// the zero-argument case because concat() is not legal. This should never be
// called directly.
template <typename Arg> void concatImpl(raw_ostream &os, Arg &&arg) {
  os << arg;
}

// Helper function for @ref concat. This should not be called directly.
template <typename Arg, typename... Args>
void concatImpl(raw_ostream &os, Arg &&arg, Args &&...args) {
  os << arg;
  concatImpl(os, args...);
}

/// Serialize the given arguments to a string and return the concatentation of
/// these string representations. The arguments must be serializable using an
/// llvm::raw_ostream. This is only really intended to construct names using a
/// combination of std::string, llvm::StringRef and integers. In principle, it
/// could be used for other things too.
///
/// FIXME: Conditionally enable this template only if an overload of operator<<
/// exists for the type.
template <typename Arg, typename... Args>
std::string concat(Arg &&arg, Args &&...args) {
  std::string buf;
  raw_string_ostream os(buf);

  concatImpl(os, arg, args...);
  return os.str();
}

/// Render the given type as a string. This is most often used to handle the
/// values of command line arguments, but it could also be used for other
/// things.
///
/// FIXME: Conditionally enable this template only if an overload of operator<<
/// exists for the type.
template <typename T> std::string toString(const T &Val) {
  std::string Buf;
  llvm::raw_string_ostream Os(Buf);
  Os << Val;

  return Os.str();
}

} // namespace tapir

} // namespace llvm

#endif // LLVM_TAPIR_STRING_UTILS_H

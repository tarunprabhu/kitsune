//===- ToString.h - Conversion functions from strings ----------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions to parse Kitsune-specific types from their string representations.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_SUPPORT_FROM_STRING_H
#define KITSUNE_SUPPORT_FROM_STRING_H

#include "llvm/ADT/StringRef.h"

#include <optional>

namespace llvm {

/// \addtogroup kitsune
/// @{

/// Parse a string to a value of the given type. If the string could not be
/// parsed, return std::nullopt.
template <typename T> std::optional<T> fromString(StringRef s);

/// Parse a string to a value of the given type. If the string could not be
/// parsed, return std::nullopt.
template <typename T> std::optional<T> fromString(const char *cstr) {
  return fromString<T>(StringRef(cstr));
}

/// @}

} // namespace llvm

#endif // KITSUNE_SUPPORT_FROM_STRING_H

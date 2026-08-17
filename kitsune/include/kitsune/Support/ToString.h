//===- ToString.h - Convert Kitsune-specific types to string ---*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to convert Kitsune-specific types to string. This could be used to
// convert other types to string as well.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_SUPPORT_TO_STRING_H
#define KITSUNE_SUPPORT_TO_STRING_H

#include "kitsune/Support/TypeTraits.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

/// \addtogroup kitsune
/// @{

template <int N> std::string toString(const char (&s)[N]) { return s; }

template <typename T,
          std::enable_if_t<std::is_same_v<T, const char *>, int> = 0>
std::string toString(T v);

template <typename T,
          std::enable_if_t<std::is_arithmetic_v<T> || std::is_enum_v<T> ||
                               std::is_string_like_v<T>,
                           int> = 0>
std::string toString(const T &v);

template <typename T,
          std::enable_if_t<std::is_iterable_v<T> && !std::is_string_like_v<T>,
                           int> = 0>
std::string toString(const T &container, StringRef sep = ",") {
  std::string buf;
  raw_string_ostream os(buf);

  bool comma = false;
  for (const auto &v : container) {
    if (comma)
      os << sep;
    os << llvm::toString(v);
    comma = true;
  }
  os.flush();

  return buf;
}

/// Convert the name of the type to a string suitable for printing. For example,
/// int32_t will be rendered to the string "int32_t".
template <typename T, std::enable_if_t<std::is_scalar_v<T>, int> = 0>
StringRef toString();

template <typename T, std::enable_if_t<std::is_same_v<T, std::string>, int> = 0>
StringRef toString() {
  return "std::string";
}

template <typename T,
          std::enable_if_t<std::is_same_v<T, llvm::StringRef>, int> = 0>
StringRef toString() {
  return "StringRef";
}

template <typename T, std::enable_if_t<std::is_small_vector_v<T>, int> = 0>
StringRef toString() {
  return "SmallVector";
}

template <typename T, std::enable_if_t<std::is_small_set_v<T>, int> = 0>
StringRef toString() {
  return "SmallSet";
}

/// @}

} // namespace llvm

#endif // KITSUNE_SUPPORT_TO_STRING_H

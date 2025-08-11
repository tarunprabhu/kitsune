//===- StringUtils.h - String utilities ------------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// String utilities for kitsune. Some of these are tweaks to the utilities
// provided by LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_SUPPORT_STRING_UTILS_H
#define KITSUNE_SUPPORT_STRING_UTILS_H

#include "llvm/Support/raw_ostream.h"

namespace llvm {

namespace detail {

void sjoin(raw_ostream &os);

template <typename T, typename... Args>
void sjoin(raw_ostream &os, T &&first, Args &&...rest) {
  os << first;
  detail::sjoin(os, rest...);
}

} // namespace detail

/// Concatenate the given arguments into a string. All the arguments provided
/// must be serializable using the << operator and LLVM's raw_ostream.
template <typename T, typename... Args>
std::string sjoin(T &&first, Args &&...rest) {
  std::string buf;
  raw_string_ostream os(buf);
  detail::sjoin(os, first, rest...);
  return os.str();
}

} // namespace llvm

#endif // KITSUNE_SUPPORT_STRING_UTILS_H

//===- VerifyImpl.h - Utilities for Kitsune's verifiers --------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities shared between Kitsune's verifiers.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_CORE_VERIFY_IMPL_H
#define KITSUNE_LIB_CORE_VERIFY_IMPL_H

#include "llvm/Support/FormatVariadic.h"

namespace llvm {

namespace detail {

/// If \cond is not true, and the optional output stream \p os has been
/// provided, print an error message to \p os. Return \p cond.
template <typename... T>
bool check(bool cond, raw_ostream *os, StringRef fmt, T &&...args) {
  if (!cond && os)
    (*os) << formatv(fmt.data(), args...) << "\n";
  return cond;
}

} // namespace detail

} // namespace llvm

#endif // KITSUNE_LIB_CORE_VERIFY_IMPL_H

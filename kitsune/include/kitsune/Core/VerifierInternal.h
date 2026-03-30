//===- VerifierInternal.h - Utilities for Kitsune's verifiers --*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private header containing utilities for Kitsune's verifiers.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_VERIFIER_INTERNAL_H
#define KITSUNE_CORE_VERIFIER_INTERNAL_H

#include "llvm/Support/FormatVariadic.h"

namespace llvm {

namespace detail {

static constexpr StringRef errMsgAttrBadValues =
    "Unexpected number of values '{}' (expected '{}') in attribute '{}'";

static constexpr StringRef errMsgAttrNoValue =
    "Could not get value of type '{}' in attribute '{}'";

static constexpr StringRef errMsgAttrNoValueAt =
    "Could not get value of type '{}' at index '{}' in attribute '{}'";

static constexpr StringRef errMsgAttrValue =
    "invalid value for attribute '{}'. {}";

static constexpr StringRef errMsgAttrIncompatible =
    "attributes '{}' and '{}' are incompatible";

} // namespace detail

/// If \cond is not true, and the optional output stream \p os has been
/// provided, print an error message to \p os. Return \p cond.
template <typename... T>
bool Verify(bool cond, raw_ostream *os, StringRef fmt, T &&...args) {
  if (!cond && os)
    (*os) << formatv(fmt.data(), args...) << "\n";
  return cond;
}

} // namespace llvm

#endif // KITSUNE_CORE_VERIFIER_INTERNAL_H

//===- Verifier.h - Interface for Kitsune-specific verifiers ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Kitsune-specific verifiers
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_VERIFIER_H
#define KITSUNE_CORE_VERIFIER_H

namespace llvm {

class Function;
class Module;
class raw_ostream;

/// \addtogroup kitsune
/// @{

/// Verify an LLVM function. This returns true if no errors were found, false
/// otherwise. This is intentionally the opposite of how LLVM's standard
/// verifier behaves since it is more "natural". If an optional output stream is
/// provided, error messages will be written to it. If \p kitsuneOnly is `true`,
/// only Kitsune-specific verification will be carried out. Otherwise, this
/// will call's LLVM's standard verifier to perform both Kitsune-specific and
/// the standard LLVM verification.
bool verifyFunction(const Function &f, bool kitsuneOnly, raw_ostream *os);

//// Verify an LLVM module. This returns true if no errors were found, false
/// otherwise. This is intentionally the opposite of how LLVM's standard
/// verifier behaves since it is more "natural". If an optional output stream is
/// provided, error messages will be written to it. If \p kitsuneOnly is `true`,
/// only Kitsune-specific verification will be carried out. Otherwise, this
/// will call's LLVM's standard verifier to perform both Kitsune-specific and
/// the standard LLVM verification.
bool verifyModule(const Module &m, bool kitsuneOnly, raw_ostream *os);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_VERIFIER_H

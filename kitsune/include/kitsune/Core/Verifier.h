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

#include "kitsune/Support/Diagnostics.h"
#include "llvm/ADT/SmallVector.h"

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

/// A (mostly) opaque verifier object. This only exposes the check methods that
/// are used by the custom attribute verification functions. The only entry
/// points to run this verifier are in the implementations of the top-level
/// \ref verifyFunction and \ref verifyModule functions.
class KitVerifier {
private:
  raw_ostream *os = nullptr;
  bool res = true;

  // The same verifier is used to verify every attribute. We cannot just return
  // call the isOk method to determine whether verification succeeded for a
  // given attribute since verification of a previous attribute may have failed.
  // Instead, we push a new value before beginning verification of an attribute
  // and accumulate the result of the verification at the end.
  SmallVector<bool, 4> stack;

protected:
  void verify(const Argument &a);
  void verify(const Function &f);
  void verify(const GlobalAlias &g);
  void verify(const GlobalIFunc &g);
  void verify(const GlobalVariable &g);
  void verify(const Instruction &inst);
  void verify(const Module &m);

  void setFailed() {
    if (stack.empty())
      res &= false;
    else
      stack.back() = false;
  }

public:
  KitVerifier() = default;
  KitVerifier(raw_ostream *os) : os(os) {}
  ~KitVerifier() = default;

  void push() { stack.push_back(true); }
  bool pop() {
    res &= stack.back();
    return stack.pop_back_val();
  }

  bool result() const { return res; }
  raw_ostream *getOstream() const { return os; }

  /// Check that the condition \p cond is true. If it is not, and an output
  /// stream has been set, emit the diagnostic \p id. \p args are additional
  /// diagnostics required by \p id.
  template <typename... Args> bool check(bool cond, DiagID id, Args &&...args) {
    if (!cond) {
      if (os)
        emitDiagnosticTo(*os, id, args...);
      setFailed();
    }
    return cond;
  }

  /// Check that the condition \p cond is true. If it is not, and an output
  /// stream has been set, emit the diagnostic \p id. \p args are additional
  /// diagnostics required by \p id. \p e is the IR element that issues the
  /// diagnostic. This must be an Argument, Function, GlobalVariable,
  /// Instruction, Loop, or Module.
  template <typename IRElem, typename... Args>
  bool check(bool cond, const IRElem &e, DiagID id, Args &&...args) {
    if (!cond) {
      if (os)
        emitDiagnosticTo(*os, e, id, args...);
      setFailed();
    }
    return cond;
  }

  friend bool verifyFunction(const Function &f, bool kitOnly, raw_ostream *os);
  friend bool verifyModule(const Module &m, bool kitOnly, raw_ostream *os);
};

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_VERIFIER_H

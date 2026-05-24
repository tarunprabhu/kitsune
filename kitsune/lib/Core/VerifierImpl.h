//=- VerifierImpl.h - Implementation of Kitsune-specific verifier -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private header for the implementation of Kitsune-specific verifiers.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_CORE_VERIFIER_IMPL_H
#define KITSUNE_LIB_CORE_VERIFIER_IMPL_H

#include "kitsune/Core/Diagnostics.h"

namespace llvm {

class Argument;
class Function;
class GlobalAlias;
class GlobalIFunc;
class GlobalVariable;
class Instruction;
class Module;
class raw_ostream;

/// A (mostly) opaque verifier object. This only exposes the check methods that
/// are used by the custom attribute verification functions. The only entry
/// points to run this verifier are in the implementations of the top-level
/// \ref verifyFunction and \ref verifyModule functions.
class KitVerifier {
private:
  raw_ostream *os = nullptr;
  bool res = true;

protected:
  KitVerifier &verifyIntrReduce(const CallBase &call, Value *unit,
                                Value *reducer, unsigned extraArgNum);
  KitVerifier &verifyIntrReduce0(const CallBase &call);
  KitVerifier &verifyIntrReduce1(const CallBase &call);
  KitVerifier &verify(const CallBase &call);

  KitVerifier &verify(const Argument &a);
  KitVerifier &verify(const Function &f);
  KitVerifier &verify(const GlobalAlias &g);
  KitVerifier &verify(const GlobalIFunc &g);
  KitVerifier &verify(const GlobalVariable &g);
  KitVerifier &verify(const Instruction &inst);
  KitVerifier &verify(const Module &m);

public:
  KitVerifier() = default;
  KitVerifier(raw_ostream *os) : os(os) {}
  ~KitVerifier() = default;

  bool result() const { return res; }
  raw_ostream *getOstream() const { return os; }

  /// Check that the condition \p cond is true. If it is not, and an output
  /// stream has been set, emit a diagnostic. \p args are passed as is to the
  /// diagnostic emission functions and should be valid. Always returns \p cond.
  template <typename... Args> bool check(bool cond, Args &&...args) {
    if (!cond) {
      if (os)
        emitDiagnosticTo(*os, args...);
      res = false;
    }
    return cond;
  }

  friend bool verifyFunction(const Function &f, bool kitOnly, raw_ostream *os);
  friend bool verifyModule(const Module &m, bool kitOnly, raw_ostream *os);
};

} // namespace llvm

#endif // KITSUNE_LIB_CORE_VERIFIER_IMPL_H

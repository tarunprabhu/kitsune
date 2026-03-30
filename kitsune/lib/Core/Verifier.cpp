//===- Verifier.cpp - Interface for Kitsune-specific verifiers ------------===//
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

#include "kitsune/Core/Verifier.h"
#include "kitsune/Core/FuncAttrs.h"
#include "kitsune/Core/GVAttrs.h"
#include "kitsune/Core/InstAttrs.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/ModuleAttrs.h"
#include "kitsune/Core/VerifierInternal.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"

using namespace llvm;

void KitVerifier::verify(const Argument &a) {
  // Argument attributes have not been implemented. When they are, they should
  // be verified.
}

void KitVerifier::verify(const Function &f) {
  for (const Argument &a : f.args())
    verify(a);

  for (const_inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i)
    verify(*i);

  for (const MDNode &attr : attrs(f)) {
    StringRef attrName = getRawAttrName(attr);
    verifyAttr(*this, f, *getFuncAttrKind(attrName));
  }
}

void KitVerifier::verify(const GlobalAlias &g) {
  // Nothing Kitsune-specific to be done here.
}

void KitVerifier::verify(const GlobalIFunc &g) {
  // Nothing Kitsune-specific to be done here.
}

void KitVerifier::verify(const GlobalVariable &g) {
  for (const MDNode &attr : attrs(g)) {
    StringRef attrName = getRawAttrName(attr);
    verifyAttr(*this, g, *getGVAttrKind(attrName));
  }
}

void KitVerifier::verify(const Instruction &inst) {
  for (const MDNode &attr : attrs(inst)) {
    StringRef attrName = getRawAttrName(attr);
    verifyAttr(*this, inst, *getInstAttrKind(attrName));
  }
}

void KitVerifier::verify(const Module &m) {
  for (const Function &f : m.functions())
    if (f.size())
      verify(f);

  for (const GlobalAlias &g : m.aliases())
    verify(g);

  for (const GlobalIFunc &ifunc : m.ifuncs())
    verify(ifunc);

  for (const GlobalVariable &g : m.globals())
    verify(g);

  for (const MDNode &attr : attrs(m)) {
    StringRef attrName = getRawAttrName(attr);
    verifyAttr(*this, m, *getModuleAttrKind(attrName));
  }

  // Some checks of "related" attributes cannot be reasonably added to the
  // verifier of either attribute. Do those here.

  // There can be at most one global variable containing device code per
  // tapir target.
  SmallDenseMap<TTID, unsigned> dcGlobals(4);
  for (const GlobalVariable &g : m.globals())
    if (std::optional<TTID> tt = getDeviceCodeAttr(g))
      ++dcGlobals[*tt];

  for (const auto &[tt, n] : dcGlobals)
    check(n <= 1, DiagID::ErrTooManyDeviceCodeGlobals, tt);

  // If a global variable containing bitcode exists, then a corresponding global
  // containing device code must also exist. The reverse is not true. Once the
  // device code has been generated, the global containing bitcode is removed.
  SmallDenseMap<TTID, unsigned> bcGlobals(4);
  for (const GlobalVariable &g : m.globals())
    if (std::optional<TTID> tt = getBitCodeAttr(g))
      ++bcGlobals[*tt];

  for (const auto &[tt, n] : bcGlobals) {
    check(n <= 1, DiagID::ErrTooManyBitCodeGlobals, tt);
    check(dcGlobals.contains(tt), DiagID::ErrMissingDeviceCodeGlobal, tt);
  }
}

bool llvm::verifyFunction(const Function &f, bool kitOnly, raw_ostream *os) {
  if (!kitOnly)
    // LLVM's verifyFunction will call this function with kitOnly == true, so
    // Kitsune-specific verification will be performed then. But the value
    // returned by verifyFunction will be the opposite of what this function
    // should return.
    return !verifyFunction(f, os);

  KitVerifier v(os);
  v.verify(f);
  return v.result();
}

bool llvm::verifyModule(const Module &m, bool kitOnly, raw_ostream *os) {
  if (!kitOnly)
    // LLVM's verifyModule will call this function with kitOnly == true, so
    // Kitsune-specific verification will be performed then. But the value
    // returned by verifyFunction will be the opposite of what this function
    // should return.
    return !verifyModule(m, os);

  KitVerifier v(os);
  v.verify(m);
  return v.result();
}

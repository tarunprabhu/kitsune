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
#include "AttrsImpl.h"
#include "ArgAttrsImpl.h"
#include "FuncAttrsImpl.h"
#include "GVAttrsImpl.h"
#include "InstAttrsImpl.h"
#include "LoopAttrsImpl.h"
#include "ModuleAttrsImpl.h"
#include "VerifierImpl.h"
#include "kitsune/Core/Attrs.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"

using namespace llvm;

KitVerifier &KitVerifier::verify(const Argument &a) {
  for (const MDNode &attr : detail::attrs(a))
    detail::verifyAttr(*this, a, detail::getRawAttrName(attr));
  return *this;
}

KitVerifier &KitVerifier::verify(const Function &f) {
  for (const Argument &a : f.args())
    verify(a);

  for (const_inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i)
    verify(*i);

  DominatorTree dt(const_cast<Function&>(f));
  LoopInfo li(dt);
  for (const Loop *loop : li)
    for (const MDNode &attr : detail::attrs(*loop))
      detail::verifyAttr(*this, *loop, detail::getRawAttrName(attr));

  return *this;
}

KitVerifier &KitVerifier::verify(const GlobalAlias &g) {
  // Nothing Kitsune-specific to be done here.
  return *this;
}

KitVerifier &KitVerifier::verify(const GlobalIFunc &g) {
  // Nothing Kitsune-specific to be done here.
  return *this;
}

KitVerifier &KitVerifier::verify(const GlobalVariable &g) {
  for (const MDNode &attr : detail::attrs(g))
    detail::verifyAttr(*this, g, detail::getRawAttrName(attr));
  return *this;
}

KitVerifier &KitVerifier::verify(const Instruction &inst) {
  for (const MDNode &attr : detail::attrs(inst))
    detail::verifyAttr(*this, inst, detail::getRawAttrName(attr));
  return *this;
}

KitVerifier &KitVerifier::verify(const Module &m) {
  for (const Function &f : m.functions())
    if (f.size())
      verify(f);

  for (const GlobalAlias &g : m.aliases())
    verify(g);

  for (const GlobalIFunc &ifunc : m.ifuncs())
    verify(ifunc);

  for (const GlobalVariable &g : m.globals())
    verify(g);

  for (const MDNode &attr : detail::attrs(m))
    detail::verifyAttr(*this, m, detail::getRawAttrName(attr));

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

  return *this;
}

bool llvm::verifyFunction(const Function &f, bool kitOnly, raw_ostream *os) {
  if (kitOnly)
    return KitVerifier(os).verify(f).result();

  // LLVM's verifyFunction will call this function with kitOnly == true, so
  // Kitsune-specific verification will be performed then. But the value
  // returned by llvm::verifyFunction will be the opposite of what this function
  // should return.
  return !verifyFunction(f, os);
}

bool llvm::verifyModule(const Module &m, bool kitOnly, raw_ostream *os) {
  if (kitOnly)
    return KitVerifier(os).verify(m).result();

  // LLVM's verifyModule will call this function with kitOnly == true, so
  // Kitsune-specific verification will be performed then. But the value
  // returned by llvm::verifyModule will be the opposite of what this function
  // should return.
  return !verifyModule(m, os);
}

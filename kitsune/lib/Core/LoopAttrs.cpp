//===- LoopAttrs.cpp - Kitsune-specific loop attributes and utilities -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions and utilities to work with tapir-specific attributes (really
// LLVM-IR metadata) on tapir loops.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/LoopAttrs.h"
#include "AttrsImpl.h"
#include "LoopAttrsImpl.h"
#include "VerifierImpl.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Support/Diagnostics.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Module.h"

using namespace llvm;

//------------------------------------------------------------------------------

MDNode *llvm::detail::getRawAttrList(const Loop &loop) {
  return loop.getLoopID();
}

void llvm::detail::setAttrList(Loop &loop, MDNode *attrList) {
  return loop.setLoopID(attrList);
}

void llvm::detail::addAttr(Loop &loop, StringRef name,
                           ArrayRef<Metadata *> vals) {
  LLVMContext &ctx = getContext(loop);
  MDNode *attrList = getRawAttrList(loop);
  MDNode *newAttrList = getAttrListWith(name, vals, attrList, ctx);

  setAttrList(loop, newAttrList);
}

void llvm::detail::removeAttr(Loop &loop, StringRef attrName) {
  MDNode *attrList = getRawAttrList(loop);
  MDNode *newAttrList = getAttrListWithout(attrName, attrList);

  setAttrList(loop, newAttrList);
}

void llvm::detail::verifyAttr(KitVerifier &v, const Loop &loop,
                              StringRef attrName) {
#define LOOP_ATTR(NAME, IRNAME, ...)                                           \
  if (attrName == IRNAME)                                                      \
    return verify##NAME##Attr(v, loop);
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

//------------------------------------------------------------------------------

raw_ostream &llvm::operator<<(raw_ostream &os, const LoopAttrKind &attr) {
  return os << getAttrName(attr);
}

StringRef llvm::getAttrName(LoopAttrKind attr) {
  switch (attr) {
#define LOOP_ATTR(NAME, IRNAME, ...)                                           \
  case LoopAttrKind::NAME:                                                     \
    return IRNAME;
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
  }
  llvm_unreachable("getAttrName: Attribute not handled");
}

std::optional<LoopAttrKind> llvm::getLoopAttrKind(StringRef name) {
  return StringSwitch<std::optional<LoopAttrKind>>(name)
#define LOOP_ATTR(NAME, IRNAME, ...) .Case(IRNAME, LoopAttrKind::NAME)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
      .Default(std::nullopt);
}

void llvm::addAttr(Loop &loop, LoopAttrKind attr) {
  switch (attr) {
  default:
    emitDiagnostic(DiagID::ErrAttrAdd, getAttrName(attr));
    exitOnError();
    break;
#define LOOP_ATTR_0(NAME, IRNAME, ...)                                         \
  case LoopAttrKind::NAME:                                                     \
    return detail::addAttr(loop, IRNAME, {});
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
  }
}

DEFN_ATTR_GENERIC(Loop, LoopAttrKind)

#define LOOP_ATTR(...) DEFN_ATTR_COMMON(Loop, LoopAttrKind, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

#define LOOP_ATTR_LOOP(...) DEFN_ATTR_LOOP(Loop, __VA_ARGS__)
#define LOOP_ATTR_0(...) DEFN_ATTR_0(Loop, __VA_ARGS__)
#define LOOP_ATTR_1(...) DEFN_ATTR_1(Loop, __VA_ARGS__)
#define LOOP_ATTR_2(...) DEFN_ATTR_2(Loop, __VA_ARGS__)
#define LOOP_ATTR_3(...) DEFN_ATTR_3(Loop, __VA_ARGS__)
#define LOOP_ATTR_4(...) DEFN_ATTR_4(Loop, __VA_ARGS__)
#define LOOP_ATTR_5(...) DEFN_ATTR_5(Loop, __VA_ARGS__)
#define LOOP_ATTR_6(...) DEFN_ATTR_6(Loop, __VA_ARGS__)
#define LOOP_ATTR_7(...) DEFN_ATTR_7(Loop, __VA_ARGS__)
#define LOOP_ATTR_8(...) DEFN_ATTR_8(Loop, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

#define LOOP_ATTR_N(...) DEFN_ATTR_N(Loop, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

// -----------------------------------------------------------------------------
// Add custom attribute verifiers here. In general, you should not modify
// anything above this line unless you are modifying a core part of the
// attribute implementation.

void llvm::verifyNameAttr(KitVerifier &v, const Loop &loop,
                          const StringRef &name) {
  v.check(name.size(), loop, DiagID::ErrAttrBadValue, LoopAttrKind::Name,
          DiagMessage::errEmptyStr);
}

void llvm::verifyThreadsPerBlockAttr(KitVerifier &v, const Loop &loop,
                                     const int32_t &tpb) {
  v.check(tpb >= 0 && tpb <= 1024, loop, DiagID::ErrAttrBadValue,
          LoopAttrKind::ThreadsPerBlock, "Must be in the range [0,1024]");
}

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
#include "kitsune/Core/AttrsCommon.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/MetadataUtils.h"
#include "kitsune/Support/Diagnostics.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Module.h"

using namespace llvm;

static void setAttrList(Loop &loop, MDNode *attrList) {
  return loop.setLoopID(attrList);
}

static void addAttr(Loop &loop, StringRef name, ArrayRef<Metadata *> vals) {
  LLVMContext &ctx = getContext(loop);
  MDNode *attrList = getAttrList(loop);
  MDNode *newAttrList = getNewAttrListWith(name, vals, attrList, ctx);

  setAttrList(loop, newAttrList);
}

static void removeAttr(Loop &loop, StringRef attrName) {
  MDNode *attrList = getAttrList(loop);
  MDNode *newAttrList = getNewAttrListWithout(attrName, attrList);

  setAttrList(loop, newAttrList);
}

MDNode *llvm::getAttrList(const Loop &loop) { return loop.getLoopID(); }

StringRef llvm::getAttrName(LoopAttrKind attr) {
  switch (attr) {
#define LOOP_ATTR(NAME, IRNAME, TYPE)                                          \
  case LoopAttrKind::NAME:                                                     \
    return IRNAME;
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
  }
  llvm_unreachable("getAttrName: Attribute not handled");
}

std::optional<LoopAttrKind> llvm::getLoopAttrKind(StringRef name) {
  return StringSwitch<std::optional<LoopAttrKind>>(name)
#define LOOP_ATTR(NAME, IRNAME, TYPE) .Case(IRNAME, LoopAttrKind::NAME)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
      .Default(std::nullopt);
}

bool llvm::verifyAttr(const Loop &loop, LoopAttrKind attr, raw_ostream *os) {
  switch (attr) {
#define LOOP_ATTR(NAME, IRNAME, TYPE)                                          \
  case LoopAttrKind::NAME:                                                     \
    return verify##NAME##Attr(loop, os);
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
  }
  llvm_unreachable("verifyAttr: Attribute not handled");
}

bool llvm::hasAttr(const Loop &loop, LoopAttrKind attr) {
  return getRawAttr(getAttrName(attr), getAttrList(loop));
}

void llvm::addAttr(Loop &loop, LoopAttrKind attr) {
  StringRef attrName = getAttrName(attr);
  switch (attr) {
  default:
    emitDiagnostic(DiagID::ErrAttrWithoutValues, attrName);
    exitOnError();
    break;
#define LOOP_ATTR_0(NAME, IRNAME) case LoopAttrKind::NAME:
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
    return ::addAttr(loop, attrName, {});
  }
}

void llvm::removeAttr(Loop &loop, LoopAttrKind attr) {
  ::removeAttr(loop, getAttrName(attr));
}

#define LOOP_ATTR(NAME, IRNAME, TYPE)                                          \
  bool llvm::has##NAME##Attr(const Loop &loop) {                               \
    return getRawAttr(IRNAME, getAttrList(loop));                              \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(Loop &loop) { ::removeAttr(loop, IRNAME); }

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

#define LOOP_ATTR_LOOP(NAME, IRNAME)                                           \
  std::optional<Loop *> llvm::get##NAME##Attr(                                 \
      const Loop &loop, const SmallVectorImpl<const LoopInfo *> &lis) {        \
    return getAttrValue(IRNAME, getAttrList(loop), lis);                       \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Loop &loop, const Loop &l) {                      \
    ::addAttr(loop, IRNAME, l.getLoopID());                                    \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Loop &loop, raw_ostream *os) {           \
    return verifyAttrLoop(IRNAME, getAttrList(loop), os);                      \
  }

#define LOOP_ATTR_0(NAME, IRNAME)                                              \
  void llvm::add##NAME##Attr(Loop &loop) { ADD_0(IRNAME, loop); }              \
                                                                               \
  bool llvm::verify##NAME##Attr(const Loop &loop, raw_ostream *os) {           \
    return verifyAttr0(IRNAME, getAttrList(loop), os);                         \
  }

#define LOOP_ATTR_1(NAME, IRNAME, TYPE)                                        \
  std::optional<TYPE> llvm::get##NAME##Attr(const Loop &loop) {                \
    return getAttrValue<TYPE>(IRNAME, getAttrList(loop), 0, 1);                \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Loop &loop, TYPE val) {                           \
    ADD_1(IRNAME, loop, val);                                                  \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Loop &loop, raw_ostream *os) {           \
    VERIFY_1(os, loop, NAME, IRNAME, TYPE);                                    \
  }

#define LOOP_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)        \
  void llvm::add##NAME##Attr(Loop &loop, ETY0 e0, ETY1 e1) {                   \
    ADD_2(IRNAME, loop, e0, e1);                                               \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Loop &loop, raw_ostream *os) {           \
    VERIFY_2(os, loop, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1);              \
  }

#define LOOP_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2)                                               \
  void llvm::add##NAME##Attr(Loop &loop, ETY0 e0, ETY1 e1, ETY2 e2) {          \
    ADD_3(IRNAME, loop, e0, e1, e2);                                           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Loop &loop, raw_ostream *os) {           \
    VERIFY_3(os, loop, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2,         \
             ENAME2);                                                          \
  }

#define LOOP_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3)                            \
  void llvm::add##NAME##Attr(Loop &loop, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3) { \
    ADD_4(IRNAME, loop, e0, e1, e2, e3);                                       \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Loop &loop, raw_ostream *os) {           \
    VERIFY_4(os, loop, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2, \
             ETY3, ENAME3);                                                    \
  }

#define LOOP_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4)         \
  void llvm::add##NAME##Attr(Loop &loop, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,   \
                             ETY4 e4) {                                        \
    ADD_5(IRNAME, loop, e0, e1, e2, e3, e4);                                   \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Loop &loop, raw_ostream *os) {           \
    VERIFY_5(os, loop, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2, \
             ETY3, ENAME3, ETY4, ENAME4);                                      \
  }

#define LOOP_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5)                                               \
  void llvm::add##NAME##Attr(Loop &loop, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,   \
                             ETY4 e4, ETY5 e5) {                               \
    ADD_6(IRNAME, loop, e0, e1, e2, e3, e4, e5);                               \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Loop &loop, raw_ostream *os) {           \
    VERIFY_6(os, loop, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2, \
             ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5);                        \
  }

#define LOOP_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6)                            \
  void llvm::add##NAME##Attr(Loop &loop, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,   \
                             ETY4 e4, ETY5 e5, ETY6 e6) {                      \
    ADD_7(IRNAME, loop, e0, e1, e2, e3, e4, e5, e6);                           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Loop &loop, raw_ostream *os) {           \
    VERIFY_7(os, loop, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2, \
             ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5, ETY6, ENAME6);          \
  }

#define LOOP_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7)         \
  void llvm::add##NAME##Attr(Loop &loop, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,   \
                             ETY4 e4, ETY5 e5, ETY6 e6, ETY7 e7) {             \
    ADD_8(IRNAME, loop, e0, e1, e2, e3, e4, e5, e6, e7);                       \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Loop &loop, raw_ostream *os) {           \
    VERIFY_8(os, loop, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2, \
             ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5, ETY6, ENAME6, ETY7,     \
             ENAME7);                                                          \
  }

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

#define LOOP_ATTR_N(NAME, IRNAME, ETY, ENAME, EN, NELEMS)                      \
  std::optional<ETY> llvm::get##ENAME##From##NAME##Attr(const Loop &loop) {    \
    return getAttrValue<ETY>(IRNAME, getAttrList(loop), EN, NELEMS);           \
  }
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

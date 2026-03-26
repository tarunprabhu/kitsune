//===- GVAttrs.cpp - Kitsune-specific attributes for global variables -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions and utilities to work with Kitsune-specific attributes for global
// variables.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/GVAttrs.h"
#include "kitsune/Core/AttrsCommon.h"
#include "kitsune/Core/GVUtils.h"
#include "kitsune/Core/MetadataUtils.h"
#include "kitsune/Support/Diagnostics.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/GlobalVariable.h"

using namespace llvm;

static void setAttrList(GlobalVariable &g, MDNode *attrList) {
  g.setMetadata(LLVMContext::MD_kit_gv_attrs, attrList);
}

static void addAttr(GlobalVariable &g, StringRef name,
                    ArrayRef<Metadata *> vals) {
  LLVMContext &ctx = g.getContext();
  MDNode *attrList = getAttrList(g);
  MDNode *newAttrList = getNewAttrListWith(name, vals, attrList, ctx);

  setAttrList(g, newAttrList);
}

static void removeAttr(GlobalVariable &g, StringRef attrName) {
  MDNode *attrList = getAttrList(g);
  MDNode *newAttrList = getNewAttrListWithout(attrName, attrList);

  setAttrList(g, newAttrList);
}

MDNode *llvm::getAttrList(const GlobalVariable &g) {
  return g.getMetadata(LLVMContext::MD_kit_gv_attrs);
}

StringRef llvm::getAttrName(GVAttrKind attr) {
  switch (attr) {
#define GV_ATTR(NAME, IRNAME, TYPE)                                            \
  case GVAttrKind::NAME:                                                       \
    return IRNAME;
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
  }
  llvm_unreachable("getAttrName: Attribute not handled");
}

std::optional<GVAttrKind> llvm::getGVAttrKind(StringRef name) {
  return StringSwitch<std::optional<GVAttrKind>>(name)
#define GV_ATTR(NAME, IRNAME, TYPE) .Case(IRNAME, GVAttrKind::NAME)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
      .Default(std::nullopt);
}

bool llvm::verifyAttr(const GlobalVariable &g, GVAttrKind attr,
                      raw_ostream *os) {
  switch (attr) {
#define GV_ATTR(NAME, IRNAME, TYPE)                                            \
  case GVAttrKind::NAME:                                                       \
    return verify##NAME##Attr(g, os);
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
  }
  llvm_unreachable("verifyAttr: Attribute not handled");
}

bool llvm::hasAttr(const GlobalVariable &g, GVAttrKind attr) {
  return getRawAttr(getAttrName(attr), getAttrList(g));
}

void llvm::addAttr(GlobalVariable &g, GVAttrKind attr) {
  StringRef attrName = getAttrName(attr);
  switch (attr) {
  default:
    emitDiagnostic(DiagID::ErrAttrWithoutValues, attrName);
    exitOnError();
    break;
#define GV_ATTR_0(NAME, IRNAME) case GVAttrKind::NAME:
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
    return ::addAttr(g, attrName, {});
  }
}

void llvm::removeAttr(GlobalVariable &g, GVAttrKind attr) {
  ::removeAttr(g, getAttrName(attr));
}

#define GV_ATTR(NAME, IRNAME, TYPE)                                            \
  bool llvm::has##NAME##Attr(const GlobalVariable &g) {                        \
    return getRawAttr(IRNAME, getAttrList(g));                                 \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(GlobalVariable &g) { ::removeAttr(g, IRNAME); }

#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

#define GV_ATTR_LOOP(NAME, IRNAME)                                             \
  std::optional<Loop *> llvm::get##NAME##Attr(                                 \
      const GlobalVariable &g, const SmallVectorImpl<const LoopInfo *> &lis) { \
    return getAttrValue(IRNAME, getAttrList(g), lis);                          \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(GlobalVariable &g, const Loop &loop) {            \
    ::addAttr(g, IRNAME, loop.getLoopID());                                    \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g, raw_ostream *os) {    \
    return verifyAttrLoop(IRNAME, getAttrList(g), os);                         \
  }

#define GV_ATTR_0(NAME, IRNAME)                                                \
  void llvm::add##NAME##Attr(GlobalVariable &g) { ADD_0(IRNAME, g); }          \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g, raw_ostream *os) {    \
    return verifyAttr0(IRNAME, getAttrList(g), os);                            \
  }

#define GV_ATTR_1(NAME, IRNAME, TYPE)                                          \
  std::optional<TYPE> llvm::get##NAME##Attr(const GlobalVariable &g) {         \
    return ::getAttrValue<TYPE>(IRNAME, getAttrList(g), 0, 1);                 \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(GlobalVariable &g, TYPE val) {                    \
    ADD_1(IRNAME, g, val);                                                     \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g, raw_ostream *os) {    \
    VERIFY_1(os, g, NAME, IRNAME, TYPE);                                       \
  }

#define GV_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)          \
  void llvm::add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1) {            \
    ADD_2(IRNAME, g, e0, e1);                                                  \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g, raw_ostream *os) {    \
    VERIFY_2(os, g, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1);                 \
  }

#define GV_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2)                                                 \
  void llvm::add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1, ETY2 e2) {   \
    ADD_3(IRNAME, g, e0, e1, e2);                                              \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g, raw_ostream *os) {    \
    VERIFY_3(os, g, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2);   \
  }

#define GV_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2, ETY3, ENAME3, EN3)                              \
  void llvm::add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1, ETY2 e2,     \
                             ETY3 e3) {                                        \
    ADD_4(IRNAME, g, e0, e1, e2, e3);                                          \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g, raw_ostream *os) {    \
    VERIFY_4(os, g, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2,    \
             ETY3, ENAME3);                                                    \
  }

#define GV_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4)           \
  void llvm::add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1, ETY2 e2,     \
                             ETY3 e3, ETY4 e4) {                               \
    ADD_5(IRNAME, g, e0, e1, e2, e3, e4);                                      \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g, raw_ostream *os) {    \
    VERIFY_5(os, g, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2,    \
             ETY3, ENAME3, ETY4, ENAME4);                                      \
  }

#define GV_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,     \
                  ENAME5, EN5)                                                 \
  void llvm::add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1, ETY2 e2,     \
                             ETY3 e3, ETY4 e4, ETY5 e5) {                      \
    ADD_6(IRNAME, g, e0, e1, e2, e3, e4, e5);                                  \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g, raw_ostream *os) {    \
    VERIFY_6(os, g, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2,    \
             ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5);                        \
  }

#define GV_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,     \
                  ENAME5, EN5, ETY6, ENAME6, EN6)                              \
  void llvm::add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1, ETY2 e2,     \
                             ETY3 e3, ETY4 e4, ETY5 e5, ETY6 e6) {             \
    ADD_7(IRNAME, g, e0, e1, e2, e3, e4, e5, e6);                              \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g, raw_ostream *os) {    \
    VERIFY_7(os, g, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2,    \
             ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5, ETY6, ENAME6);          \
  }

#define GV_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,     \
                  ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7)           \
  void llvm::add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1, ETY2 e2,     \
                             ETY3 e3, ETY4 e4, ETY5 e5, ETY6 e6, ETY7 e7) {    \
    ADD_8(IRNAME, g, e0, e1, e2, e3, e4, e5, e6, e7);                          \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g, raw_ostream *os) {    \
    VERIFY_8(os, g, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2,    \
             ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5, ETY6, ENAME6, ETY7,     \
             ENAME7);                                                          \
  }

#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

#define GV_ATTR_N(NAME, IRNAME, ETY, ENAME, EN, NELEMS)                        \
  std::optional<ETY> llvm::get##ENAME##From##NAME##Attr(                       \
      const GlobalVariable &g) {                                               \
    return getAttrValue<ETY>(IRNAME, getAttrList(g), EN, NELEMS);              \
  }
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

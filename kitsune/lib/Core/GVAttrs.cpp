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
#include "llvm/IR/GlobalVariable.h"

using namespace llvm;

static void addAttr(GlobalVariable &g, GVAttrKind attr,
                    ArrayRef<Metadata *> ops) {
  LLVMContext &ctx = g.getContext();
  StringRef attrName = getAttrName(attr);
  MDNode *md = MDNode::get(ctx, ops);

  removeAttr(g, attr);
  g.addMetadata(attrName, *md);
}

template <typename T>
static std::optional<T> getAttr(const GlobalVariable &g, GVAttrKind attr,
                                unsigned i, unsigned n) {
  StringRef attrName = getAttrName(attr);
  if (MDNode *md = g.getMetadata(attrName))
    if (md->getNumOperands() == n)
      return fromMetadata<T>(md->getOperand(i));
  return std::nullopt;
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
  return g.hasMetadata(getAttrName(attr));
}

void llvm::addAttr(GlobalVariable &g, GVAttrKind attr) {
  switch (attr) {
  default:
    emitDiagnostic(DiagID::ErrAttrWithoutValues, getAttrName(attr));
    exitOnError();
    break;
#define GV_ATTR_0(NAME, IRNAME) case GVAttrKind::NAME:
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
    return ::addAttr(g, attr, {});
  }
}

void llvm::removeAttr(GlobalVariable &g, GVAttrKind attr) {
  g.setMetadata(getAttrName(attr), nullptr);
}

#define GV_ATTR(NAME, IRNAME, TYPE)                                            \
  bool llvm::has##NAME##Attr(const GlobalVariable &g) {                        \
    return hasAttr(g, GVAttrKind::NAME);                                       \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(GlobalVariable &g) {                           \
    removeAttr(g, GVAttrKind::NAME);                                           \
  }

#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

#define GV_ATTR_0(NAME, IRNAME)                                                \
  void llvm::add##NAME##Attr(GlobalVariable &g) {                              \
    ADD_0(GVAttrKind, NAME, g);                                                \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g, raw_ostream *os) {    \
    if (MDNode *md = g.getMetadata(IRNAME))                                    \
      VERIFY_0(md->getNumOperands() == 0, IRNAME, os);                         \
    return true;                                                               \
  }

#define GV_ATTR_1(NAME, IRNAME, TYPE)                                          \
  std::optional<TYPE> llvm::get##NAME##Attr(const GlobalVariable &g) {         \
    return getAttr<TYPE>(g, GVAttrKind::NAME, 0, 1);                           \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(GlobalVariable &g, TYPE val) {                    \
    ADD_1(GVAttrKind, NAME, g, val);                                           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g, raw_ostream *os) {    \
    VERIFY_1(os, g, NAME, IRNAME, TYPE);                                       \
  }

#define GV_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)          \
  void llvm::add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1) {            \
    ADD_2(GVAttrKind, NAME, g, e0, e1);                                        \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g, raw_ostream *os) {    \
    VERIFY_2(os, g, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1);                 \
  }

#define GV_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2)                                                 \
  void llvm::add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1, ETY2 e2) {   \
    ADD_3(GVAttrKind, NAME, g, e0, e1, e2);                                    \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g, raw_ostream *os) {    \
    VERIFY_3(os, g, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2);   \
  }

#define GV_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2, ETY3, ENAME3, EN3)                              \
  void llvm::add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1, ETY2 e2,     \
                             ETY3 e3) {                                        \
    ADD_4(GVAttrKind, NAME, g, e0, e1, e2, e3);                                \
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
    ADD_5(GVAttrKind, NAME, g, e0, e1, e2, e3, e4);                            \
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
    ADD_6(GVAttrKind, NAME, g, e0, e1, e2, e3, e4, e5);                        \
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
    ADD_7(GVAttrKind, NAME, g, e0, e1, e2, e3, e4, e5, e6);                    \
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
    ADD_8(GVAttrKind, NAME, g, e0, e1, e2, e3, e4, e5, e6, e7);                \
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
    return getAttr<ETY>(g, GVAttrKind::NAME, EN, NELEMS);                      \
  }
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

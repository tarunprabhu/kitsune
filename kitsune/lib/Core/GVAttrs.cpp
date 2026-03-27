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
#include "AttrsImpl.h"
#include "kitsune/Core/AttrsCommon.h"
#include "kitsune/Core/GVUtils.h"
#include "kitsune/Support/Diagnostics.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/GlobalVariable.h"

using namespace llvm;

static void setAttrList(GlobalVariable &g, MDNode *attrList) {
  g.setMetadata(LLVMContext::MD_kit_gv_attrs, attrList);
}

static void addAttr(GlobalVariable &g, StringRef name,
                    ArrayRef<Metadata *> vals) {
  LLVMContext &ctx = g.getContext();
  MDNode *attrList = getAttrList(g);
  MDNode *newAttrList = getAttrListWith(name, vals, attrList, ctx);

  setAttrList(g, newAttrList);
}

static void removeAttr(GlobalVariable &g, StringRef attrName) {
  MDNode *attrList = getAttrList(g);
  MDNode *newAttrList = getAttrListWithout(attrName, attrList);

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

void llvm::addAttr(GlobalVariable &g, GVAttrKind attr) {
  switch (attr) {
  default:
    emitDiagnostic(DiagID::ErrAttrWithoutValues, getAttrName(attr));
    exitOnError();
    break;
#define GV_ATTR_0(NAME, IRNAME)                                                \
  case GVAttrKind::NAME:                                                       \
    return ::addAttr(g, IRNAME, {});
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
  }
}

DEFN_ATTR_GENERIC(GlobalVariable, GVAttrKind)

#define GV_ATTR(...) DEFN_ATTR_COMMON(GlobalVariable, GVAttrKind, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

#define GV_ATTR_LOOP(...) DEFN_ATTR_LOOP(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_0(...) DEFN_ATTR_0(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_1(...) DEFN_ATTR_1(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_2(...) DEFN_ATTR_2(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_3(...) DEFN_ATTR_3(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_4(...) DEFN_ATTR_4(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_5(...) DEFN_ATTR_5(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_6(...) DEFN_ATTR_6(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_7(...) DEFN_ATTR_7(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_8(...) DEFN_ATTR_8(GlobalVariable, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

#define GV_ATTR_N(...) DEFN_ATTR_N(GlobalVariable, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

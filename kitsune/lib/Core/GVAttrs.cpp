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
#include "kitsune/Core/MetadataUtils.h"
#include "kitsune/Support/Diagnostics.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/GlobalVariable.h"

using namespace llvm;

template <typename T>
static void addAttrAs(GlobalVariable &g, GVAttrKind attr, T val) {
  removeAttr(g, attr);

  LLVMContext &ctx = g.getContext();
  StringRef attrName = getAttrName(attr);
  MDNode *md = MDNode::get(ctx, {toMetadata(val, ctx)});
  g.addMetadata(attrName, *md);
  llvm::errs() << g << "\n";
}

static void addAttr(GlobalVariable &g, GVAttrKind attr) {
  LLVMContext &ctx = g.getContext();
  StringRef attrName = getAttrName(attr);
  MDNode *md = MDNode::get(ctx, {});
  g.addMetadata(attrName, *md);
}

template <typename T>
static std::optional<T> getAttr(const GlobalVariable &g, GVAttrKind attr) {
  StringRef attrName = getAttrName(attr);
  if (MDNode *md = g.getMetadata(attrName))
    if (md->getNumOperands() == 1)
      return fromMetadata<T>(md->getOperand(0));
  return std::nullopt;
}

StringRef llvm::getAttrName(GVAttrKind attr) {
  switch (attr) {
#define GV_ATTR(NAME, TYPE, IRNAME)                                            \
  case GVAttrKind::NAME:                                                       \
    return IRNAME;
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
  }
  llvm_unreachable("getAttrName: Attribute not handled");
}

std::optional<GVAttrKind> llvm::getGVAttrKind(StringRef name) {
  return StringSwitch<std::optional<GVAttrKind>>(name)
#define GV_ATTR(NAME, TYPE, IRNAME) .Case(IRNAME, GVAttrKind::NAME)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
      .Default(std::nullopt);
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
#define GV_ATTR_FLAG(NAME, IRNAME) case GVAttrKind::NAME:
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
    return ::addAttr(g, attr);
  }
}

void llvm::removeAttr(GlobalVariable &g, GVAttrKind attr) {
  g.setMetadata(getAttrName(attr), nullptr);
}

// Flag attributes (those that do not have a value), have a different set of
// accessors. Mask these before generating declarations for the other
// attributes.
#define GV_ATTR_FLAG(NAME, IRNAME)
#define GV_ATTR(NAME, TYPE, IRNAME)                                            \
  bool llvm::has##NAME##Attr(const GlobalVariable &g) {                        \
    return hasAttr(g, GVAttrKind::NAME);                                       \
  }                                                                            \
                                                                               \
  std::optional<TYPE> llvm::get##NAME##Attr(const GlobalVariable &g) {         \
    return getAttr<TYPE>(g, GVAttrKind::NAME);                                 \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(GlobalVariable &g, TYPE val) {                    \
    addAttrAs(g, GVAttrKind::NAME, val);                                       \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(GlobalVariable &g) {                           \
    removeAttr(g, GVAttrKind::NAME);                                           \
  }
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

#define GV_ATTR_FLAG(NAME, IRNAME)                                             \
  bool llvm::has##NAME##Attr(const GlobalVariable &g) {                        \
    return hasAttr(g, GVAttrKind::NAME);                                       \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(GlobalVariable &g) {                              \
    addAttr(g, GVAttrKind::NAME);                                              \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(GlobalVariable &g) {                           \
    removeAttr(g, GVAttrKind::NAME);                                           \
  }
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

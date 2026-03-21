//===- FuncAttrs.cpp - Kitsune-specific attributes for functions ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions and utilities to work with function attributes. These are not
// known to LLVM.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/FuncAttrs.h"
#include "kitsune/Core/MetadataUtils.h"
#include "kitsune/Support/Diagnostics.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Function.h"

using namespace llvm;

template <typename T>
static void addAttrAs(Function &f, FuncAttrKind attr, T val) {
  LLVMContext &ctx = f.getContext();
  StringRef attrName = getAttrName(attr);
  MDNode *md = MDNode::get(ctx, {toMetadata(val, ctx)});
  f.addMetadata(attrName, *md);
}

static void addAttr(Function &f, FuncAttrKind attr) {
  LLVMContext &ctx = f.getContext();
  StringRef attrName = getAttrName(attr);
  MDNode *md = MDNode::get(ctx, {});
  f.addMetadata(attrName, *md);
}

template <typename T>
static std::optional<T> getAttr(const Function &f, FuncAttrKind attr) {
  StringRef attrName = getAttrName(attr);
  if (MDNode *md = f.getMetadata(attrName))
    if (md->getNumOperands() == 1)
      return fromMetadata<T>(md->getOperand(0));
  return std::nullopt;
}

StringRef llvm::getAttrName(FuncAttrKind attr) {
  switch (attr) {
#define FUNC_ATTR(NAME, TYPE, IRNAME)                                          \
  case FuncAttrKind::NAME:                                                     \
    return IRNAME;
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
  }
  llvm_unreachable("getAttrName: Attribute not handled");
}

std::optional<FuncAttrKind> llvm::getFuncAttrKind(StringRef name) {
  return StringSwitch<std::optional<FuncAttrKind>>(name)
#define FUNC_ATTR(NAME, TYPE, IRNAME) .Case(IRNAME, FuncAttrKind::NAME)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
      .Default(std::nullopt);
}

bool llvm::hasAttr(const Function &f, FuncAttrKind attr) {
  return f.hasMetadata(getAttrName(attr));
}

void llvm::addAttr(Function &f, FuncAttrKind attr) {
  switch (attr) {
  default:
    emitDiagnostic(DiagID::ErrAttrWithoutValues, getAttrName(attr));
    exitOnError();
    break;
#define FUNC_ATTR_FLAG(NAME, IRNAME) case FuncAttrKind::NAME:
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
    return ::addAttr(f, attr);
  }
}

void llvm::removeAttr(Function &f, FuncAttrKind attr) {
  f.setMetadata(getAttrName(attr), nullptr);
}

// Flag attributes (those that do not have a value), have a different set of
// accessors. Mask these before generating declarations for the other
// attributes.
#define FUNC_ATTR_FLAG(NAME, IRNAME)
#define FUNC_ATTR(NAME, TYPE, IRNAME)                                          \
  bool llvm::has##NAME##Attr(const Function &f) {                              \
    return hasAttr(f, FuncAttrKind::NAME);                                     \
  }                                                                            \
                                                                               \
  std::optional<TYPE> llvm::get##NAME##Attr(const Function &f) {               \
    return getAttr<TYPE>(f, FuncAttrKind::NAME);                               \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Function &f, TYPE val) {                          \
    addAttrAs(f, FuncAttrKind::NAME, val);                                     \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(Function &f) {                                 \
    removeAttr(f, FuncAttrKind::NAME);                                         \
  }
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

#define FUNC_ATTR_FLAG(NAME, IRNAME)                                           \
  bool llvm::has##NAME##Attr(const Function &f) {                              \
    return hasAttr(f, FuncAttrKind::NAME);                                     \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Function &f) { addAttr(f, FuncAttrKind::NAME); }  \
                                                                               \
  void llvm::remove##NAME##Attr(Function &f) {                                 \
    removeAttr(f, FuncAttrKind::NAME);                                         \
  }
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

//===- ModuleAttrs.cpp - Module attributes and utilities ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions and utilities to work with module "attributes" (really named
// LLVM-IR metadata).
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ModuleAttrs.h"
#include "AttrsImpl.h"
#include "kitsune/Core/AttrsCommon.h"
#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Support/Diagnostics.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Module.h"

using namespace llvm;

static void setAttrList(Module &m, MDNode *attrList) {
  NamedMDNode *nmd = m.getOrInsertNamedMetadata("kit.module");
  if (nmd->getNumOperands())
    nmd->setOperand(0, attrList);
  else
    nmd->addOperand(attrList);
}

static void addAttr(Module &m, StringRef name, ArrayRef<Metadata *> vals) {
  LLVMContext &ctx = m.getContext();
  MDNode *attrList = getAttrList(m);
  MDNode *newAttrList = getAttrListWith(name, vals, attrList, ctx);

  setAttrList(m, newAttrList);
}

static void removeAttr(Module &m, StringRef attrName) {
  MDNode *attrList = getAttrList(m);
  MDNode *newAttrList = getAttrListWithout(attrName, attrList);

  setAttrList(m, newAttrList);
}

MDNode *llvm::getAttrList(const Module &m) {
  if (NamedMDNode *nmd = m.getNamedMetadata("kit.module"))
    if (nmd->getNumOperands())
      return nmd->getOperand(0);
  return nullptr;
}

StringRef llvm::getAttrName(ModuleAttrKind attrKind) {
  switch (attrKind) {
#define MODULE_ATTR(NAME, IRNAME, TYPE)                                        \
  case ModuleAttrKind::NAME:                                                   \
    return IRNAME;
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
  }
  llvm_unreachable("getAttrName: ModuleAttrKind not handled");
}

std::optional<ModuleAttrKind> llvm::getModuleAttrKind(StringRef name) {
  return StringSwitch<std::optional<ModuleAttrKind>>(name)
#define MODULE_ATTR(NAME, IRNAME, TYPE) .Case(IRNAME, ModuleAttrKind::NAME)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
      .Default(std::nullopt);
}

bool llvm::verifyAttr(const Module &m, ModuleAttrKind attr, raw_ostream *os) {
  switch (attr) {
#define MODULE_ATTR(NAME, IRNAME, TYPE)                                        \
  case ModuleAttrKind::NAME:                                                   \
    return verify##NAME##Attr(m, os);
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
  }
  llvm_unreachable("verifyAttr: Attribute not handled");
}

void llvm::addAttr(Module &m, ModuleAttrKind attr) {
  switch (attr) {
  default:
    emitDiagnostic(DiagID::ErrAttrWithoutValues, getAttrName(attr));
    exitOnError();
    break;
#define MODULE_ATTR_0(NAME, IRNAME)                                            \
  case ModuleAttrKind::NAME:                                                   \
    return ::addAttr(m, IRNAME, {});
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
  }
}

DEFN_ATTR_GENERIC(Module, ModuleAttrKind)

#define MODULE_ATTR(...) DEFN_ATTR_COMMON(Module, ModuleAttrKind, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

#define MODULE_ATTR_LOOP(...) DEFN_ATTR_LOOP(Module, __VA_ARGS__)
#define MODULE_ATTR_0(...) DEFN_ATTR_0(Module, __VA_ARGS__)
#define MODULE_ATTR_1(...) DEFN_ATTR_1(Module, __VA_ARGS__)
#define MODULE_ATTR_2(...) DEFN_ATTR_2(Module, __VA_ARGS__)
#define MODULE_ATTR_3(...) DEFN_ATTR_3(Module, __VA_ARGS__)
#define MODULE_ATTR_4(...) DEFN_ATTR_4(Module, __VA_ARGS__)
#define MODULE_ATTR_5(...) DEFN_ATTR_5(Module, __VA_ARGS__)
#define MODULE_ATTR_6(...) DEFN_ATTR_6(Module, __VA_ARGS__)
#define MODULE_ATTR_7(...) DEFN_ATTR_7(Module, __VA_ARGS__)
#define MODULE_ATTR_8(...) DEFN_ATTR_8(Module, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

#define MODULE_ATTR_N(...) DEFN_ATTR_N(Module, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

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
#include "ModuleAttrsImpl.h"
#include "VerifierImpl.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Core/TTUtils.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Module.h"

using namespace llvm;

//------------------------------------------------------------------------------

MDNode *llvm::detail::getRawAttrList(const Module &m) {
  if (NamedMDNode *nmd = m.getNamedMetadata("kit.module"))
    if (nmd->getNumOperands())
      return nmd->getOperand(0);
  return nullptr;
}

void llvm::detail::setAttrList(Module &m, MDNode *attrList) {
  NamedMDNode *nmd = m.getOrInsertNamedMetadata("kit.module");
  if (nmd->getNumOperands())
    nmd->setOperand(0, attrList);
  else
    nmd->addOperand(attrList);
}

void llvm::detail::addAttr(Module &m, StringRef name,
                           ArrayRef<Metadata *> vals) {
  LLVMContext &ctx = m.getContext();
  MDNode *attrList = getRawAttrList(m);
  MDNode *newAttrList = getAttrListWith(name, vals, attrList, ctx);

  setAttrList(m, newAttrList);
}

void llvm::detail::removeAttr(Module &m, StringRef attrName) {
  MDNode *attrList = getRawAttrList(m);
  MDNode *newAttrList = getAttrListWithout(attrName, attrList);

  setAttrList(m, newAttrList);
}

void llvm::detail::verifyAttr(KitVerifier &v, const Module &m,
                              StringRef attrName) {
#define MODULE_ATTR(NAME, IRNAME, ...)                                         \
  if (attrName == IRNAME)                                                      \
    return verify##NAME##Attr(v, m);
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

//------------------------------------------------------------------------------

raw_ostream &llvm::operator<<(raw_ostream &os, const ModuleAttrKind &attr) {
  return os << getAttrName(attr);
}

StringRef llvm::getAttrName(ModuleAttrKind attrKind) {
  switch (attrKind) {
#define MODULE_ATTR(NAME, IRNAME, ...)                                         \
  case ModuleAttrKind::NAME:                                                   \
    return IRNAME;
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
  }
  llvm_unreachable("getAttrName: ModuleAttrKind not handled");
}

std::optional<ModuleAttrKind> llvm::getModuleAttrKind(StringRef name) {
  return StringSwitch<std::optional<ModuleAttrKind>>(name)
#define MODULE_ATTR(NAME, IRNAME, ...) .Case(IRNAME, ModuleAttrKind::NAME)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
      .Default(std::nullopt);
}

void llvm::addAttr(Module &m, ModuleAttrKind attr) {
  switch (attr) {
  default:
    emitDiagnostic(DiagID::ErrAttrAdd, getAttrName(attr));
    exitOnError();
    break;
#define MODULE_ATTR_0(NAME, IRNAME, ...)                                       \
  case ModuleAttrKind::NAME:                                                   \
    return detail::addAttr(m, IRNAME, {});
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
  }
}

DEFN_ATTR_GENERIC(Module, ModuleAttrKind)

#define MODULE_ATTR(...) DEFN_ATTR_COMMON(Module, ModuleAttrKind, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

#define MODULE_ATTR_L(...) DEFN_ATTR_L(Module, __VA_ARGS__)
#define MODULE_ATTR_S(...) DEFN_ATTR_S(Module, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

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

// -----------------------------------------------------------------------------
// Add custom attribute verifiers here. In general, you should not modify
// anything above this line unless you are modifying a core part of the
// attribute implementation.

void llvm::verifyDeviceModuleFlagsAttr(KitVerifier &v, const Module &m,
                                       const TTID &tt, const StringRef &name) {
  ModuleAttrKind attr = ModuleAttrKind::DeviceModuleFlags;

  v.check(generatesEmbBC(tt), DiagID::ErrAttrBadValueAt, attr, 0,
          DiagMessage::errTTEmbBC);
  v.check(name.size(), DiagID::ErrAttrBadValueAt, attr, 1,
          DiagMessage::errEmptyStr);
}

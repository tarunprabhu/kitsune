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
#include "AttrsImpl.h"
#include "kitsune/Core/FuncUtils.h"
#include "kitsune/Core/Verifier.h"
#include "kitsune/Support/Diagnostics.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Function.h"

using namespace llvm;

static void setAttrList(Function &f, MDNode *attrList) {
  f.setMetadata(LLVMContext::MD_kit_func_attrs, attrList);
}

static void addAttr(Function &f, StringRef name, ArrayRef<Metadata *> vals) {
  LLVMContext &ctx = f.getContext();
  MDNode *attrList = getRawAttrList(f);
  MDNode *newAttrList = getAttrListWith(name, vals, attrList, ctx);

  setAttrList(f, newAttrList);
}

static void removeAttr(Function &f, StringRef attrName) {
  MDNode *attrList = getRawAttrList(f);
  MDNode *newAttrList = getAttrListWithout(attrName, attrList);

  setAttrList(f, newAttrList);
}

raw_ostream &llvm::operator<<(raw_ostream &os, const FuncAttrKind &attr) {
  return os << getAttrName(attr);
}

MDNode *llvm::getRawAttrList(const Function &f) {
  return f.getMetadata(LLVMContext::MD_kit_func_attrs);
}

StringRef llvm::getAttrName(FuncAttrKind attr) {
  switch (attr) {
#define FUNC_ATTR(NAME, IRNAME, ...)                                           \
  case FuncAttrKind::NAME:                                                     \
    return IRNAME;
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
  }
  llvm_unreachable("getAttrName: Attribute not handled");
}

std::optional<FuncAttrKind> llvm::getFuncAttrKind(StringRef name) {
  return StringSwitch<std::optional<FuncAttrKind>>(name)
#define FUNC_ATTR(NAME, IRNAME, ...) .Case(IRNAME, FuncAttrKind::NAME)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
      .Default(std::nullopt);
}

bool llvm::verifyAttr(KitVerifier &v, const Function &f, FuncAttrKind attr) {
  switch (attr) {
#define FUNC_ATTR(NAME, IRNAME, ...)                                           \
  case FuncAttrKind::NAME:                                                     \
    return verify##NAME##Attr(v, f);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
  }
  llvm_unreachable("verifyAttr: Attribute not handled");
}

void llvm::addAttr(Function &f, FuncAttrKind attr) {
  switch (attr) {
  default:
    emitDiagnostic(DiagID::ErrAttrAdd, getAttrName(attr));
    exitOnError();
    break;
#define FUNC_ATTR_0(NAME, IRNAME, ...)                                         \
  case FuncAttrKind::NAME:                                                     \
    return ::addAttr(f, IRNAME, {});
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
  }
}

DEFN_ATTR_GENERIC(Function, FuncAttrKind)

#define FUNC_ATTR(...) DEFN_ATTR_COMMON(Function, FuncAttrKind, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

#define FUNC_ATTR_LOOP(...) DEFN_ATTR_LOOP(Function, __VA_ARGS__)
#define FUNC_ATTR_0(...) DEFN_ATTR_0(Function, __VA_ARGS__)
#define FUNC_ATTR_1(...) DEFN_ATTR_1(Function, __VA_ARGS__)
#define FUNC_ATTR_2(...) DEFN_ATTR_2(Function, __VA_ARGS__)
#define FUNC_ATTR_3(...) DEFN_ATTR_3(Function, __VA_ARGS__)
#define FUNC_ATTR_4(...) DEFN_ATTR_4(Function, __VA_ARGS__)
#define FUNC_ATTR_5(...) DEFN_ATTR_5(Function, __VA_ARGS__)
#define FUNC_ATTR_6(...) DEFN_ATTR_6(Function, __VA_ARGS__)
#define FUNC_ATTR_7(...) DEFN_ATTR_7(Function, __VA_ARGS__)
#define FUNC_ATTR_8(...) DEFN_ATTR_8(Function, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

#define FUNC_ATTR_N(...) DEFN_ATTR_N(Function, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

// -----------------------------------------------------------------------------
// Add custom attribute verifiers here. In general, you should not modify
// anything above this line unless you are modifying a core part of the
// attribute implementation.

bool llvm::verifyDeviceAttr(KitVerifier &v, const Function &f,
                            const bool &hasAttr) {
  bool ok = true;
  FuncAttrKind attr = FuncAttrKind::Device;

  ok &= v.check(!hasKernelAttr(f), f, DiagID::ErrAttrNotCompatible, attr,
                FuncAttrKind::Kernel);

  return ok;
}

bool llvm::verifyKernelAttr(KitVerifier &v, const Function &f,
                            const bool &hasAttr) {
  bool ok = true;
  FuncAttrKind attr = FuncAttrKind::Kernel;

  ok &= v.check(!hasDeviceAttr(f), f, DiagID::ErrAttrNotCompatible, attr,
                FuncAttrKind::Device);

  return ok;
}

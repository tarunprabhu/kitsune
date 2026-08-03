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
#include "FuncAttrsImpl.h"
#include "VerifierImpl.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/FuncUtils.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Function.h"

using namespace llvm;

//------------------------------------------------------------------------------

MDNode *llvm::detail::getRawAttrList(const Function &f) {
  return f.getMetadata(LLVMContext::MD_kit_func_attrs);
}

void llvm::detail::setAttrList(Function &f, MDNode *attrList) {
  f.setMetadata(LLVMContext::MD_kit_func_attrs, attrList);
}

void llvm::detail::addAttr(Function &f, StringRef name,
                           ArrayRef<Metadata *> vals) {
  LLVMContext &ctx = f.getContext();
  MDNode *attrList = getRawAttrList(f);
  MDNode *newAttrList = getAttrListWith(name, vals, attrList, ctx);

  setAttrList(f, newAttrList);
}

void llvm::detail::removeAttr(Function &f, StringRef attrName) {
  MDNode *attrList = getRawAttrList(f);
  MDNode *newAttrList = getAttrListWithout(attrName, attrList);

  setAttrList(f, newAttrList);
}

void llvm::detail::verifyAttr(KitVerifier &v, const Function &f,
                              StringRef attrName) {
#define FUNC_ATTR(NAME, IRNAME, ...)                                           \
  if (attrName == IRNAME)                                                      \
    return verify##NAME##Attr(v, f);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

//------------------------------------------------------------------------------

raw_ostream &llvm::operator<<(raw_ostream &os, const FuncAttrKind &attr) {
  return os << getAttrName(attr);
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

void llvm::addAttr(Function &f, FuncAttrKind attr) {
  switch (attr) {
  default:
    emitDiagnostic(DiagID::ErrAttrAdd, getAttrName(attr));
    exitOnError();
    break;
#define FUNC_ATTR_0(NAME, IRNAME, ...)                                         \
  case FuncAttrKind::NAME:                                                     \
    return detail::addAttr(f, IRNAME, {});
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
  }
}

DEFN_ATTR_GENERIC(Function, FuncAttrKind)

#define FUNC_ATTR(...) DEFN_ATTR_COMMON(Function, FuncAttrKind, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

#define FUNC_ATTR_L(...) DEFN_ATTR_L(Function, __VA_ARGS__)
#define FUNC_ATTR_S(...) DEFN_ATTR_S(Function, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

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

void llvm::verifyDeviceAttr(KitVerifier &v, const Function &f,
                            const bool &hasAttr) {
  FuncAttrKind attr = FuncAttrKind::Device;
  v.check(!hasKernelAttr(f), f, DiagID::ErrAttrNotCompatible, attr,
          FuncAttrKind::Kernel);
}

void llvm::verifyKernelAttr(KitVerifier &v, const Function &f,
                            const bool &hasAttr) {
  FuncAttrKind attr = FuncAttrKind::Kernel;
  v.check(!hasDeviceAttr(f), f, DiagID::ErrAttrNotCompatible, attr,
          FuncAttrKind::Device);
}

//===- ArgAttrs.cpp - Kitsune-specific attributes for function arguments --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to work with attributes for function arguments. These are not known
// to LLVM.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ArgAttrs.h"
#include "ArgAttrsImpl.h"
#include "AttrsImpl.h"
#include "VerifierImpl.h"
#include "kitsune/Core/ArgUtils.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Argument.h"

using namespace llvm;

//------------------------------------------------------------------------------

MDNode *llvm::detail::getRawAttr(const Argument &a, StringRef attrName) {
  return getRawAttr(attrName, getRawAttrList(a));
}

MDNode *llvm::detail::getRawAttrList(const Argument &a) {
  assert(a.getParent() && "Argument must have a parent");

  const Function &f = *a.getParent();
  if (MDNode *allArgsList = f.getMetadata(LLVMContext::MD_kit_arg_attrs))
    if (Metadata *argAttrs = allArgsList->getOperand(a.getArgNo()).get())
      if (auto *md = dyn_cast<MDNode>(argAttrs))
        return md;
  return nullptr;
}

void llvm::detail::setAttrList(Argument &a, MDNode *attrList) {
  assert(a.getParent() && "Argument must have a parent");

  Function &f = *a.getParent();
  if (!f.getMetadata(LLVMContext::MD_kit_arg_attrs)) {
    LLVMContext &ctx = f.getContext();
    FunctionType *fty = f.getFunctionType();
    SmallVector<Metadata *, 4> ops(fty->getNumParams(), nullptr);

    f.setMetadata(LLVMContext::MD_kit_arg_attrs, MDNode::get(ctx, ops));
  }

  MDNode *allArgAttrs = f.getMetadata(LLVMContext::MD_kit_arg_attrs);
  allArgAttrs->replaceOperandWith(a.getArgNo(), attrList);
}

void llvm::detail::addAttr(Argument &a, StringRef name,
                           ArrayRef<Metadata *> vals) {
  LLVMContext &ctx = getContext(a);
  MDNode *attrList = getRawAttrList(a);
  MDNode *newAttrList = getAttrListWith(name, vals, attrList, ctx);

  setAttrList(a, newAttrList);
}

void llvm::detail::removeAttr(Argument &a, StringRef attrName) {
  MDNode *attrList = getRawAttrList(a);
  MDNode *newAttrList = getAttrListWithout(attrName, attrList);

  setAttrList(a, newAttrList);
}

void llvm::detail::verifyAttr(KitVerifier &v, const Argument &a,
                              StringRef attrName) {
#define ARG_ATTR(NAME, IRNAME, ...)                                            \
  if (attrName == IRNAME)                                                      \
    return verify##NAME##Attr(v, a);
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"
}

//------------------------------------------------------------------------------

raw_ostream &llvm::operator<<(raw_ostream &os, const ArgAttrKind &attr) {
  return os << getAttrName(attr);
}

StringRef llvm::getAttrName(ArgAttrKind attr) {
  switch (attr) {
#define ARG_ATTR(NAME, IRNAME, ...)                                            \
  case ArgAttrKind::NAME: return IRNAME;
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"
  }
  llvm_unreachable("getAttrName: Attribute not handled");
}

std::optional<ArgAttrKind> llvm::getArgAttrKind(StringRef name) {
  return StringSwitch<std::optional<ArgAttrKind>>(name)
#define ARG_ATTR(NAME, IRNAME, ...) .Case(IRNAME, ArgAttrKind::NAME)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"
      .Default(std::nullopt);
}

void llvm::addAttr(Argument &a, ArgAttrKind attr) {
  switch (attr) {
  default:
    emitDiagnostic(DiagID::ErrAttrAdd, getAttrName(attr));
    exitOnError();
    break;
#define ARG_ATTR_0(NAME, IRNAME, ...)                                          \
  case ArgAttrKind::NAME: return detail::addAttr(a, IRNAME, {});
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"
  }
}

DEFN_ATTR_GENERIC(Argument, ArgAttrKind)

#define ARG_ATTR(...) DEFN_ATTR_COMMON(Argument, ArgAttrKind, __VA_ARGS__)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"

#define ARG_ATTR_L(...) DEFN_ATTR_L(Argument, __VA_ARGS__)
#define ARG_ATTR_S(...) DEFN_ATTR_S(Argument, __VA_ARGS__)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"

#define ARG_ATTR_0(...) DEFN_ATTR_0(Argument, __VA_ARGS__)
#define ARG_ATTR_1(...) DEFN_ATTR_1(Argument, __VA_ARGS__)
#define ARG_ATTR_2(...) DEFN_ATTR_2(Argument, __VA_ARGS__)
#define ARG_ATTR_3(...) DEFN_ATTR_3(Argument, __VA_ARGS__)
#define ARG_ATTR_4(...) DEFN_ATTR_4(Argument, __VA_ARGS__)
#define ARG_ATTR_5(...) DEFN_ATTR_5(Argument, __VA_ARGS__)
#define ARG_ATTR_6(...) DEFN_ATTR_6(Argument, __VA_ARGS__)
#define ARG_ATTR_7(...) DEFN_ATTR_7(Argument, __VA_ARGS__)
#define ARG_ATTR_8(...) DEFN_ATTR_8(Argument, __VA_ARGS__)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"

#define ARG_ATTR_N(...) DEFN_ATTR_N(Argument, __VA_ARGS__)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"

// -----------------------------------------------------------------------------
// Add custom attribute verifiers here. In general, you should not modify
// anything above this line unless you are modifying a core part of the
// attribute implementation.

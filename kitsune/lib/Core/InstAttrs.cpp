//===- InstAttrs.cpp - Instruction attributes and utilities ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions and utilities to work with "attributes" (really LLVM-IR metadata)
// on instructions.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/InstAttrs.h"
#include "AttrsImpl.h"
#include "InstAttrsImpl.h"
#include "VerifierImpl.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/InstUtils.h"
#include "kitsune/Core/MetadataUtils.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

//------------------------------------------------------------------------------

MDNode *llvm::detail::getRawAttrList(const Instruction &inst) {
  return inst.getMetadata(LLVMContext::MD_kit_inst_attrs);
}

void llvm::detail::setAttrList(Instruction &inst, MDNode *attrList) {
  return inst.setMetadata(LLVMContext::MD_kit_inst_attrs, attrList);
}

void llvm::detail::addAttr(Instruction &inst, StringRef name,
                           ArrayRef<Metadata *> vals) {
  LLVMContext &ctx = inst.getContext();
  MDNode *attrList = getRawAttrList(inst);
  MDNode *newAttrList = getAttrListWith(name, vals, attrList, ctx);

  setAttrList(inst, newAttrList);
}

void llvm::detail::removeAttr(Instruction &inst, StringRef attrName) {
  MDNode *attrList = getRawAttrList(inst);
  MDNode *newAttrList = getAttrListWithout(attrName, attrList);

  setAttrList(inst, newAttrList);
}

void llvm::detail::verifyAttr(KitVerifier &v, const Instruction &inst,
                              StringRef attrName) {
#define INST_ATTR(NAME, IRNAME, ...)                                           \
  if (attrName == IRNAME)                                                      \
    return llvm::verify##NAME##Attr(v, inst);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

//------------------------------------------------------------------------------

raw_ostream &llvm::operator<<(raw_ostream &os, const InstAttrKind &attr) {
  return os << getAttrName(attr);
}

StringRef llvm::getAttrName(InstAttrKind attr) {
  switch (attr) {
#define INST_ATTR(NAME, IRNAME, ...)                                           \
  case InstAttrKind::NAME:                                                     \
    return IRNAME;
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
  }
  llvm_unreachable("getAttrName: Attribute not handled");
}

std::optional<InstAttrKind> llvm::getInstAttrKind(StringRef name) {
  return StringSwitch<std::optional<InstAttrKind>>(name)
#define INST_ATTR(NAME, IRNAME, ...) .Case(IRNAME, InstAttrKind::NAME)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
      .Default(std::nullopt);
}

void llvm::addAttr(Instruction &inst, InstAttrKind attr) {
  switch (attr) {
  default:
    emitDiagnostic(DiagID::ErrAttrAdd, getAttrName(attr));
    exitOnError();
    break;
#define INST_ATTR_0(NAME, IRNAME, ...)                                         \
  case InstAttrKind::NAME:                                                     \
    return detail::addAttr(inst, IRNAME, {});
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
  }
}

DEFN_ATTR_GENERIC(Instruction, InstAttrKind)

#define INST_ATTR(...) DEFN_ATTR_COMMON(Instruction, InstAttrKind, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_L(...) DEFN_ATTR_L(Instruction, __VA_ARGS__)
#define INST_ATTR_S(...) DEFN_ATTR_S(Instruction, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_0(...) DEFN_ATTR_0(Instruction, __VA_ARGS__)
#define INST_ATTR_1(...) DEFN_ATTR_1(Instruction, __VA_ARGS__)
#define INST_ATTR_2(...) DEFN_ATTR_2(Instruction, __VA_ARGS__)
#define INST_ATTR_3(...) DEFN_ATTR_3(Instruction, __VA_ARGS__)
#define INST_ATTR_4(...) DEFN_ATTR_4(Instruction, __VA_ARGS__)
#define INST_ATTR_5(...) DEFN_ATTR_5(Instruction, __VA_ARGS__)
#define INST_ATTR_6(...) DEFN_ATTR_6(Instruction, __VA_ARGS__)
#define INST_ATTR_7(...) DEFN_ATTR_7(Instruction, __VA_ARGS__)
#define INST_ATTR_8(...) DEFN_ATTR_8(Instruction, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_N(...) DEFN_ATTR_N(Instruction, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

// -----------------------------------------------------------------------------
// Add custom attribute verifiers here. In general, you should not modify
// anything above this line unless you are modifying a core part of the
// attribute implementation.

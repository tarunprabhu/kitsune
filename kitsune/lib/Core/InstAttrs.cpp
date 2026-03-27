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
#include "kitsune/Core/AttrsCommon.h"
#include "kitsune/Core/InstUtils.h"
#include "kitsune/Core/MetadataUtils.h"
#include "kitsune/Support/Diagnostics.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

static void setAttrList(Instruction &inst, MDNode *attrList) {
  return inst.setMetadata(LLVMContext::MD_kit_inst_attrs, attrList);
}

static void addAttr(Instruction &inst, StringRef name,
                    ArrayRef<Metadata *> vals) {
  LLVMContext &ctx = inst.getContext();
  MDNode *attrList = getAttrList(inst);
  MDNode *newAttrList = getAttrListWith(name, vals, attrList, ctx);

  setAttrList(inst, newAttrList);
}

static void removeAttr(Instruction &inst, StringRef attrName) {
  MDNode *attrList = getAttrList(inst);
  MDNode *newAttrList = getAttrListWithout(attrName, attrList);

  setAttrList(inst, newAttrList);
}

MDNode *llvm::getAttrList(const Instruction &inst) {
  return inst.getMetadata(LLVMContext::MD_kit_inst_attrs);
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

bool llvm::verifyAttr(const Instruction &inst, InstAttrKind attr,
                      raw_ostream *os) {
  switch (attr) {
#define INST_ATTR(NAME, IRNAME, ...)                                           \
  case InstAttrKind::NAME:                                                     \
    return verify##NAME##Attr(inst, os);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
  }
  llvm_unreachable("verifyAttr: Attribute not handled");
}

void llvm::addAttr(Instruction &inst, InstAttrKind attr) {
  switch (attr) {
  default:
    emitDiagnostic(DiagID::ErrAttrWithoutValues, getAttrName(attr));
    exitOnError();
    break;
#define INST_ATTR_0(NAME, IRNAME, ...)                                         \
  case InstAttrKind::NAME:                                                     \
    return ::addAttr(inst, IRNAME, {});
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
  }
}

DEFN_ATTR_GENERIC(Instruction, InstAttrKind)

#define INST_ATTR(...) DEFN_ATTR_COMMON(Instruction, InstAttrKind, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_LOOP(...) DEFN_ATTR_LOOP(Instruction, __VA_ARGS__)
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

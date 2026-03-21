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
#include "kitsune/Core/MetadataUtils.h"
#include "kitsune/Support/Diagnostics.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/LoopInfo.h"

using namespace llvm;

template <typename T>
static void addAttrAs(Instruction &inst, InstAttrKind attr, T val) {
  LLVMContext &ctx = inst.getContext();
  StringRef name = getAttrName(attr);
  MDNode *md = MDNode::get(ctx, toMetadata(val, ctx));
  inst.setMetadata(name, md);
}

template <> void addAttrAs(Instruction &inst, InstAttrKind attr, MDNode *md) {
  inst.setMetadata(getAttrName(attr), md);
}

template <>
void addAttrAs(Instruction &inst, InstAttrKind attr, const Loop *loop) {
  addAttrAs(inst, attr, loop->getLoopID());
}

static void addAttr(Instruction &inst, InstAttrKind attr) {
  LLVMContext &ctx = inst.getContext();
  inst.setMetadata(getAttrName(attr), MDNode::get(ctx, {}));
}

template <typename T>
static std::optional<T> getAttr(const Instruction &inst, InstAttrKind attr) {
  if (!hasAttr(inst, attr))
    return std::nullopt;
  MDNode *md = inst.getMetadata(getAttrName(attr));
  return fromMetadata<T>(md->getOperand(0));
}

template <>
std::optional<MDNode *> getAttr(const Instruction &inst, InstAttrKind attr) {
  if (!hasAttr(inst, attr))
    return std::nullopt;
  return inst.getMetadata(getAttrName(attr));
}

std::optional<Loop *> getAttr(const Instruction &inst, InstAttrKind attr,
                              const SmallVectorImpl<const LoopInfo *> &lis) {
  if (std::optional<MDNode *> md = getAttr<MDNode *>(inst, attr))
    for (const LoopInfo *li : lis)
      for (Loop *loop : *li)
        if (loop->getLoopID() == *md)
          return loop;
  return std::nullopt;
}

StringRef llvm::getAttrName(InstAttrKind attr) {
  switch (attr) {
#define INST_ATTR(NAME, TYPE, IRNAME)                                          \
  case InstAttrKind::NAME:                                                     \
    return IRNAME;
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
  }
  llvm_unreachable("getAttrName: Attribute not handled");
}

std::optional<InstAttrKind> llvm::getInstAttrKind(StringRef name) {
  return StringSwitch<std::optional<InstAttrKind>>(name)
#define INST_ATTR(NAME, TYPE, IRNAME) .Case(IRNAME, InstAttrKind::NAME)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
      .Default(std::nullopt);
}

bool llvm::hasAttr(const Instruction &inst, InstAttrKind attr) {
  return inst.hasMetadata(getAttrName(attr));
}

void llvm::addAttr(Instruction &inst, InstAttrKind attr) {
  switch (attr) {
  default:
    emitDiagnostic(DiagID::ErrAttrWithoutValues, getAttrName(attr));
    exitOnError();
    break;
#define INST_ATTR_FLAG(NAME, IRNAME) case InstAttrKind::NAME:
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
    return ::addAttr(inst, attr);
  }
}

void llvm::removeAttr(Instruction &inst, InstAttrKind attr) {
  inst.setMetadata(getAttrName(attr), nullptr);
}

// Flag attributes (those that do not have a value), and attributes that take
// Loop's as values will have a different set of accessors. Mask these before
// generating declarations for the other attributes.
#define INST_ATTR_FLAG(NAME, IRNAME)
#define INST_ATTR_LOOP(NAME, IRNAME)
#define INST_ATTR(NAME, TYPE, IRNAME)                                          \
  bool llvm::has##NAME##Attr(const Instruction &inst) {                        \
    return hasAttr(inst, InstAttrKind::NAME);                                  \
  }                                                                            \
                                                                               \
  std::optional<TYPE> llvm::get##NAME##Attr(const Instruction &inst) {         \
    return getAttr<TYPE>(inst, InstAttrKind::NAME);                            \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Instruction &inst, TYPE val) {                    \
    addAttrAs(inst, InstAttrKind::NAME, val);                                  \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(Instruction &inst) {                           \
    removeAttr(inst, InstAttrKind::NAME);                                      \
  }
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_FLAG(NAME, IRNAME)                                           \
  bool llvm::has##NAME##Attr(const Instruction &inst) {                        \
    return hasAttr(inst, InstAttrKind::NAME);                                  \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Instruction &inst) {                              \
    addAttr(inst, InstAttrKind::NAME);                                         \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(Instruction &inst) {                           \
    removeAttr(inst, InstAttrKind::NAME);                                      \
  }
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_LOOP(NAME, IRNAME)                                           \
  bool llvm::has##NAME##Attr(const Instruction &inst) {                        \
    return hasAttr(inst, InstAttrKind::NAME);                                  \
  }                                                                            \
                                                                               \
  std::optional<Loop *> llvm::get##NAME##Attr(                                 \
      const Instruction &inst, const SmallVectorImpl<const LoopInfo *> &lis) { \
    return getAttr(inst, InstAttrKind::NAME, lis);                             \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Instruction &inst, const Loop &loop) {            \
    addAttrAs(inst, InstAttrKind::NAME, &loop);                                \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Instruction &inst, const Loop *loop) {            \
    addAttrAs(inst, InstAttrKind::NAME, loop);                                 \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(Instruction &inst) {                           \
    removeAttr(inst, InstAttrKind::NAME);                                      \
  }
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

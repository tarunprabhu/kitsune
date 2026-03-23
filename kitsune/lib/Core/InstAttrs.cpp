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

static void addAttrImpl(Instruction &inst, InstAttrKind attr, MDNode *md) {
  StringRef attrName = getAttrName(attr);

  removeAttr(inst, attr);
  inst.setMetadata(attrName, md);
}

static void addAttr(Instruction &inst, InstAttrKind attr,
                    ArrayRef<Metadata *> ops) {
  LLVMContext &ctx = inst.getContext();
  MDNode *md = MDNode::get(ctx, ops);

  addAttrImpl(inst, attr, md);
}

static void addAttr(Instruction &inst, InstAttrKind attr, Loop &loop) {
  assert(loop.getLoopID() && "Loop does not have an ID");

  addAttrImpl(inst, attr, loop.getLoopID());
}

template <typename T>
static std::optional<T> getAttr(const Instruction &inst, InstAttrKind attr,
                                unsigned i, unsigned n) {
  StringRef attrName = getAttrName(attr);
  if (MDNode *md = inst.getMetadata(attrName))
    if (md->getNumOperands() == n)
      return fromMetadata<T>(md->getOperand(i));
  return std::nullopt;
}

static std::optional<Loop *>
getLoopAttr(const Instruction &inst, InstAttrKind attr, unsigned i, unsigned n,
            const SmallVectorImpl<const LoopInfo *> &lis) {
  StringRef attrName = getAttrName(attr);
  if (MDNode *md = inst.getMetadata(attrName))
    for (const LoopInfo *li : lis)
      for (Loop *loop : *li)
        if (loop->getLoopID() == md)
          return loop;
  return std::nullopt;
}

StringRef llvm::getAttrName(InstAttrKind attr) {
  switch (attr) {
#define INST_ATTR(NAME, IRNAME, TYPE)                                          \
  case InstAttrKind::NAME:                                                     \
    return IRNAME;
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
  }
  llvm_unreachable("getAttrName: Attribute not handled");
}

std::optional<InstAttrKind> llvm::getInstAttrKind(StringRef name) {
  return StringSwitch<std::optional<InstAttrKind>>(name)
#define INST_ATTR(NAME, IRNAME, TYPE) .Case(IRNAME, InstAttrKind::NAME)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
      .Default(std::nullopt);
}

bool llvm::verifyAttr(const Instruction &inst, InstAttrKind attr) {
  switch (attr) {
#define INST_ATTR(NAME, IRNAME, TYPE)                                          \
  case InstAttrKind::NAME:                                                     \
    return verify##NAME##Attr(inst);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
  }
  llvm_unreachable("verifyAttr: Attribute not handled");
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
#define INST_ATTR_0(NAME, IRNAME) case InstAttrKind::NAME:
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
    return ::addAttr(inst, attr, {});
  }
}

void llvm::removeAttr(Instruction &inst, InstAttrKind attr) {
  inst.setMetadata(getAttrName(attr), nullptr);
}

#define INST_ATTR(NAME, IRNAME, TYPE)                                          \
  bool llvm::has##NAME##Attr(const Instruction &inst) {                        \
    return hasAttr(inst, InstAttrKind::NAME);                                  \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(Instruction &inst) {                           \
    removeAttr(inst, InstAttrKind::NAME);                                      \
  }
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_0(NAME, IRNAME)                                              \
  void llvm::add##NAME##Attr(Instruction &inst) {                              \
    ::addAttr(inst, InstAttrKind::NAME, {});                                   \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst) {                     \
    if (MDNode *md = inst.getMetadata(IRNAME))                                 \
      return md->getNumOperands() == 0;                                        \
    return true;                                                               \
  }

#define INST_ATTR_1(NAME, IRNAME, TYPE)                                        \
  std::optional<TYPE> llvm::get##NAME##Attr(const Instruction &inst) {         \
    return getAttr<TYPE>(inst, InstAttrKind::NAME, 0, 1);                      \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Instruction &inst, TYPE val) {                    \
    LLVMContext &ctx = inst.getContext();                                      \
    Metadata *ops[] = {toMetadata(val, ctx)};                                  \
    ::addAttr(inst, InstAttrKind::NAME, ops);                                  \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst) {                     \
    if (has##NAME##Attr(inst))                                                 \
      return get##NAME##Attr(inst).has_value();                                \
    return true;                                                               \
  }

#define INST_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)        \
  void llvm::add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1) {            \
    LLVMContext &ctx = inst.getContext();                                      \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx)};              \
    ::addAttr(inst, InstAttrKind::NAME, ops);                                  \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst) {                     \
    if (has##NAME##Attr(inst))                                                 \
      return get##ENAME0##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME1##From##NAME##Attr(inst).has_value();                  \
    return true;                                                               \
  }

#define INST_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2)                                               \
  void llvm::add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1, ETY2 e2) {   \
    LLVMContext &ctx = inst.getContext();                                      \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx)};                                   \
    ::addAttr(inst, InstAttrKind::NAME, ops);                                  \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst) {                     \
    if (has##NAME##Attr(inst))                                                 \
      return get##ENAME0##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME1##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME2##From##NAME##Attr(inst).has_value();                  \
    return true;                                                               \
  }

#define INST_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3)                            \
  void llvm::add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1, ETY2 e2,     \
                             ETY3 e3) {                                        \
    LLVMContext &ctx = inst.getContext();                                      \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx), toMetadata(e3, ctx)};              \
    ::addAttr(inst, InstAttrKind::NAME, ops);                                  \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst) {                     \
    if (has##NAME##Attr(inst))                                                 \
      return get##ENAME0##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME1##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME2##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME3##From##NAME##Attr(inst).has_value();                  \
    return true;                                                               \
  }

#define INST_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4)         \
  void llvm::add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1, ETY2 e2,     \
                             ETY3 e3, ETY4 e4) {                               \
    LLVMContext &ctx = inst.getContext();                                      \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx), toMetadata(e3, ctx),               \
                       toMetadata(e4, ctx)};                                   \
    ::addAttr(inst, InstAttrKind::NAME, ops);                                  \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst) {                     \
    if (has##NAME##Attr(inst))                                                 \
      return get##ENAME0##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME1##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME2##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME3##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME4##From##NAME##Attr(inst).has_value();                  \
    return true;                                                               \
  }

#define INST_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5)                                               \
  void llvm::add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1, ETY2 e2,     \
                             ETY3 e3, ETY4 e4, ETY5 e5) {                      \
    LLVMContext &ctx = inst.getContext();                                      \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx), toMetadata(e3, ctx),               \
                       toMetadata(e4, ctx), toMetadata(e5, ctx)};              \
    ::addAttr(inst, InstAttrKind::NAME, ops);                                  \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst) {                     \
    if (has##NAME##Attr(inst))                                                 \
      return get##ENAME0##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME1##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME2##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME3##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME4##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME5##From##NAME##Attr(inst).has_value();                  \
    return true;                                                               \
  }

#define INST_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6)                            \
  void llvm::add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1, ETY2 e2,     \
                             ETY3 e3, ETY4 e4, ETY5 e5, ETY6 e6) {             \
    LLVMContext &ctx = inst.getContext();                                      \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx), toMetadata(e3, ctx),               \
                       toMetadata(e4, ctx), toMetadata(e5, ctx),               \
                       toMetadata(e6, ctx)};                                   \
    ::addAttr(inst, InstAttrKind::NAME, ops);                                  \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst) {                     \
    if (has##NAME##Attr(inst))                                                 \
      return get##ENAME0##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME1##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME2##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME3##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME4##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME5##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME6##From##NAME##Attr(inst).has_value();                  \
    return true;                                                               \
  }

#define INST_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7)         \
  void llvm::add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1, ETY2 e2,     \
                             ETY3 e3, ETY4 e4, ETY5 e5, ETY6 e6, ETY7 e7) {    \
    LLVMContext &ctx = inst.getContext();                                      \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx), toMetadata(e3, ctx),               \
                       toMetadata(e4, ctx), toMetadata(e5, ctx),               \
                       toMetadata(e6, ctx), toMetadata(e7, ctx)};              \
    ::addAttr(inst, InstAttrKind::NAME, ops);                                  \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst) {                     \
    if (has##NAME##Attr(inst))                                                 \
      return get##ENAME0##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME1##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME2##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME3##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME4##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME5##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME6##From##NAME##Attr(inst).has_value() &&                \
             get##ENAME7##From##NAME##Attr(inst).has_value();                  \
    return true;                                                               \
  }

#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_N(NAME, IRNAME, ETY, ENAME, EN, NELEMS)                      \
  std::optional<ETY> llvm::get##ENAME##From##NAME##Attr(                       \
      const Instruction &inst) {                                               \
    return getAttr<ETY>(inst, InstAttrKind::NAME, EN, NELEMS);                 \
  }
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_LOOP(NAME, IRNAME)                                           \
  std::optional<Loop *> llvm::get##NAME##Attr(                                 \
      const Instruction &inst, const SmallVectorImpl<const LoopInfo *> &lis) { \
    return getLoopAttr(inst, InstAttrKind::NAME, 0, 1, lis);                   \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Instruction &inst, Loop *loop) {                  \
    add##NAME##Attr(inst, *loop);                                              \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Instruction &inst, Loop &loop) {                  \
    ::addAttr(inst, InstAttrKind::NAME, loop);                                 \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst) {                     \
    if (MDNode *md = inst.getMetadata(IRNAME))                                 \
      return md->getNumOperands() && md->isDistinct() &&                       \
             md->getOperand(0) == md;                                          \
    return true;                                                               \
  }

#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

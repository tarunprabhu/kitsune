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
#include "kitsune/Core/AttrsCommon.h"
#include "kitsune/Core/InstructionUtils.h"
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

bool llvm::verifyAttr(const Instruction &inst, InstAttrKind attr,
                      raw_ostream *os) {
  switch (attr) {
#define INST_ATTR(NAME, IRNAME, TYPE)                                          \
  case InstAttrKind::NAME:                                                     \
    return verify##NAME##Attr(inst, os);
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
    ADD_0(InstAttrKind, NAME, inst);                                           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst, raw_ostream *os) {    \
    if (MDNode *md = inst.getMetadata(IRNAME))                                 \
      VERIFY_0(md->getNumOperands() == 0, IRNAME, os);                         \
    return true;                                                               \
  }

#define INST_ATTR_1(NAME, IRNAME, TYPE)                                        \
  std::optional<TYPE> llvm::get##NAME##Attr(const Instruction &inst) {         \
    return getAttr<TYPE>(inst, InstAttrKind::NAME, 0, 1);                      \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Instruction &inst, TYPE val) {                    \
    ADD_1(InstAttrKind, NAME, inst, val);                                      \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst, raw_ostream *os) {    \
    VERIFY_1(os, inst, NAME, IRNAME, TYPE);                                    \
  }

#define INST_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)        \
  void llvm::add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1) {            \
    ADD_2(InstAttrKind, NAME, inst, e0, e1);                                   \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst, raw_ostream *os) {    \
    VERIFY_2(os, inst, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1);              \
  }

#define INST_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2)                                               \
  void llvm::add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1, ETY2 e2) {   \
    ADD_3(InstAttrKind, NAME, inst, e0, e1, e2);                               \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst, raw_ostream *os) {    \
    VERIFY_3(os, inst, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2,         \
             ENAME2);                                                          \
  }

#define INST_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3)                            \
  void llvm::add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1, ETY2 e2,     \
                             ETY3 e3) {                                        \
    ADD_4(InstAttrKind, NAME, inst, e0, e1, e2, e3);                           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst, raw_ostream *os) {    \
    VERIFY_4(os, inst, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2, \
             ETY3, ENAME3);                                                    \
  }

#define INST_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4)         \
  void llvm::add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1, ETY2 e2,     \
                             ETY3 e3, ETY4 e4) {                               \
    ADD_5(InstAttrKind, NAME, inst, e0, e1, e2, e3, e4);                       \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst, raw_ostream *os) {    \
    VERIFY_5(os, inst, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2, \
             ETY3, ENAME3, ETY4, ENAME4);                                      \
  }

#define INST_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5)                                               \
  void llvm::add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1, ETY2 e2,     \
                             ETY3 e3, ETY4 e4, ETY5 e5) {                      \
    ADD_6(InstAttrKind, NAME, inst, e0, e1, e2, e3, e4, e5);                   \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst, raw_ostream *os) {    \
    VERIFY_6(os, inst, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2, \
             ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5);                        \
  }

#define INST_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6)                            \
  void llvm::add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1, ETY2 e2,     \
                             ETY3 e3, ETY4 e4, ETY5 e5, ETY6 e6) {             \
    ADD_7(InstAttrKind, NAME, inst, e0, e1, e2, e3, e4, e5, e6);               \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst, raw_ostream *os) {    \
    VERIFY_7(os, inst, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2, \
             ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5, ETY6, ENAME6);          \
  }

#define INST_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7)         \
  void llvm::add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1, ETY2 e2,     \
                             ETY3 e3, ETY4 e4, ETY5 e5, ETY6 e6, ETY7 e7) {    \
    ADD_8(InstAttrKind, NAME, inst, e0, e1, e2, e3, e4, e5, e6, e7);           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst, raw_ostream *os) {    \
    VERIFY_8(os, inst, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2, \
             ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5, ETY6, ENAME6, ETY7,     \
             ENAME7);                                                          \
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
  bool llvm::verify##NAME##Attr(const Instruction &inst, raw_ostream *os) {    \
    if (MDNode *md = inst.getMetadata(IRNAME)) {                               \
      if (!md->getNumOperands() || !md->isDistinct() ||                        \
          md->getOperand(0) != md) {                                           \
        if (os)                                                                \
          (*os) << "Missing value of type 'Loop' in attribute '" << IRNAME     \
                << "'\n";                                                      \
        return false;                                                          \
      }                                                                        \
    }                                                                          \
    return true;                                                               \
  }

#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

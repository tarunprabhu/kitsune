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
#include "kitsune/Core/InstUtils.h"
#include "kitsune/Core/MetadataUtils.h"
#include "kitsune/Support/Diagnostics.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/LoopInfo.h"

using namespace llvm;

static void setAttrList(Instruction &inst, MDNode *attrList) {
  return inst.setMetadata(LLVMContext::MD_kit_inst_attrs, attrList);
}

static void addAttr(Instruction &inst, StringRef name,
                    ArrayRef<Metadata *> vals) {
  LLVMContext &ctx = inst.getContext();
  MDNode *attrList = getAttrList(inst);
  MDNode *newAttrList = getNewAttrListWith(name, vals, attrList, ctx);

  setAttrList(inst, newAttrList);
}

static void removeAttr(Instruction &inst, StringRef attrName) {
  MDNode *attrList = getAttrList(inst);
  MDNode *newAttrList = getNewAttrListWithout(attrName, attrList);

  setAttrList(inst, newAttrList);
}

MDNode *llvm::getAttrList(const Instruction &inst) {
  return inst.getMetadata(LLVMContext::MD_kit_inst_attrs);
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
  return getRawAttr(getAttrName(attr), getAttrList(inst));
}

void llvm::addAttr(Instruction &inst, InstAttrKind attr) {
  StringRef attrName = getAttrName(attr);
  switch (attr) {
  default:
    emitDiagnostic(DiagID::ErrAttrWithoutValues, attrName);
    exitOnError();
    break;
#define INST_ATTR_0(NAME, IRNAME) case InstAttrKind::NAME:
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
    return ::addAttr(inst, attrName, {});
  }
}

void llvm::removeAttr(Instruction &inst, InstAttrKind attr) {
  ::removeAttr(inst, getAttrName(attr));
}

#define INST_ATTR(NAME, IRNAME, TYPE)                                          \
  bool llvm::has##NAME##Attr(const Instruction &inst) {                        \
    return getRawAttr(IRNAME, getAttrList(inst));                              \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(Instruction &inst) {                           \
    ::removeAttr(inst, IRNAME);                                                \
  }

#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_LOOP(NAME, IRNAME)                                           \
  std::optional<Loop *> llvm::get##NAME##Attr(                                 \
      const Instruction &inst, const SmallVectorImpl<const LoopInfo *> &lis) { \
    return getAttrValue(IRNAME, getAttrList(inst), lis);                       \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Instruction &inst, const Loop &loop) {            \
    ::addAttr(inst, IRNAME, loop.getLoopID());                                 \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst, raw_ostream *os) {    \
    return verifyAttrLoop(IRNAME, getAttrList(inst), os);                      \
  }

#define INST_ATTR_0(NAME, IRNAME)                                              \
  void llvm::add##NAME##Attr(Instruction &inst) { ADD_0(IRNAME, inst); }       \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst, raw_ostream *os) {    \
    return verifyAttr0(IRNAME, getAttrList(inst), os);                         \
  }

#define INST_ATTR_1(NAME, IRNAME, TYPE)                                        \
  std::optional<TYPE> llvm::get##NAME##Attr(const Instruction &inst) {         \
    return getAttrValue<TYPE>(IRNAME, getAttrList(inst), 0, 1);                \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Instruction &inst, TYPE val) {                    \
    ADD_1(IRNAME, inst, val);                                                  \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst, raw_ostream *os) {    \
    VERIFY_1(os, inst, NAME, IRNAME, TYPE);                                    \
  }

#define INST_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)        \
  void llvm::add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1) {            \
    ADD_2(IRNAME, inst, e0, e1);                                               \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Instruction &inst, raw_ostream *os) {    \
    VERIFY_2(os, inst, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1);              \
  }

#define INST_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2)                                               \
  void llvm::add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1, ETY2 e2) {   \
    ADD_3(IRNAME, inst, e0, e1, e2);                                           \
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
    ADD_4(IRNAME, inst, e0, e1, e2, e3);                                       \
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
    ADD_5(IRNAME, inst, e0, e1, e2, e3, e4);                                   \
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
    ADD_6(IRNAME, inst, e0, e1, e2, e3, e4, e5);                               \
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
    ADD_7(IRNAME, inst, e0, e1, e2, e3, e4, e5, e6);                           \
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
    ADD_8(IRNAME, inst, e0, e1, e2, e3, e4, e5, e6, e7);                       \
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
    return getAttrValue<ETY>(IRNAME, getAttrList(inst), EN, NELEMS);           \
  }
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

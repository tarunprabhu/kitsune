//===- LoopAttrs.cpp - Kitsune-specific loop attributes and utilities -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions and utilities to work with tapir-specific attributes (really
// LLVM-IR metadata) on tapir loops.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/AttrsCommon.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/MetadataUtils.h"
#include "kitsune/Support/Diagnostics.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Module.h"

using namespace llvm;

static void addAttr(Loop &loop, LoopAttrKind attr, ArrayRef<Metadata *> ops) {
  LLVMContext &ctx = getContext(loop);
  StringRef name = getAttrName(attr);
  Metadata *mdTag = MDString::get(ctx, name);

  SmallVector<Metadata *, 8> mdOps = {mdTag};
  mdOps.append(ops.begin(), ops.end());

  MDNode *md = MDNode::get(ctx, mdOps);
  MDNode *loopID = loop.getLoopID();
  MDNode *newLoopID = makePostTransformationMetadata(ctx, loopID, {name}, {md});

  loop.setLoopID(newLoopID);
}

template <typename T>
static std::optional<T> getAttr(const Loop &loop, LoopAttrKind attr, unsigned i,
                                unsigned n) {
  StringRef attrName = getAttrName(attr);
  if (MDNode *md = findOptionMDForLoop(&loop, attrName))
    if (md->getNumOperands() == n + 1)
      return fromMetadata<T>(md->getOperand(i + 1));
  return std::nullopt;
}

StringRef llvm::getAttrName(LoopAttrKind attr) {
  switch (attr) {
#define LOOP_ATTR(NAME, IRNAME, TYPE)                                          \
  case LoopAttrKind::NAME:                                                     \
    return IRNAME;
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
  }
  llvm_unreachable("getAttrName: Attribute not handled");
}

std::optional<LoopAttrKind> llvm::getLoopAttrKind(StringRef name) {
  return StringSwitch<std::optional<LoopAttrKind>>(name)
#define LOOP_ATTR(NAME, IRNAME, TYPE) .Case(IRNAME, LoopAttrKind::NAME)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
      .Default(std::nullopt);
}

bool llvm::verifyAttr(const Loop &loop, LoopAttrKind attr, raw_ostream *os) {
  switch (attr) {
#define LOOP_ATTR(NAME, IRNAME, TYPE)                                          \
  case LoopAttrKind::NAME:                                                     \
    return verify##NAME##Attr(loop, os);
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
  }
  llvm_unreachable("verifyAttr: Attribute not handled");
}

bool llvm::hasAttr(const Loop &loop, LoopAttrKind attr) {
  return findOptionMDForLoop(&loop, getAttrName(attr));
}

void llvm::addAttr(Loop &loop, LoopAttrKind attr) {
  switch (attr) {
  default:
    emitDiagnostic(DiagID::ErrAttrWithoutValues, getAttrName(attr));
    exitOnError();
    break;
#define LOOP_ATTR_0(NAME, IRNAME) case LoopAttrKind::NAME:
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
    return ::addAttr(loop, attr, {});
  }
}

void llvm::removeAttr(Loop &loop, LoopAttrKind attr) {
  LLVMContext &ctx = getContext(loop);
  StringRef name = getAttrName(attr);
  MDNode *loopID = loop.getLoopID();
  MDNode *newLoopID = makePostTransformationMetadata(ctx, loopID, {name}, {});

  loop.setLoopID(newLoopID);
}

#define LOOP_ATTR(NAME, IRNAME, TYPE)                                          \
  bool llvm::has##NAME##Attr(const Loop &loop) {                               \
    return hasAttr(loop, LoopAttrKind::NAME);                                  \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(Loop &loop) {                                  \
    removeAttr(loop, LoopAttrKind::NAME);                                      \
  }
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

#define LOOP_ATTR_0(NAME, IRNAME)                                              \
  void llvm::add##NAME##Attr(Loop &loop) { ADD_0(LoopAttrKind, NAME, loop); }  \
                                                                               \
  bool llvm::verify##NAME##Attr(const Loop &loop, raw_ostream *os) {           \
    StringRef attrName = getAttrName(LoopAttrKind::NAME);                      \
    if (MDNode *md = findOptionMDForLoop(&loop, attrName))                     \
      VERIFY_0(md->getNumOperands() == 1, IRNAME, os);                         \
    return true;                                                               \
  }

#define LOOP_ATTR_1(NAME, IRNAME, TYPE)                                        \
  std::optional<TYPE> llvm::get##NAME##Attr(const Loop &loop) {                \
    return getAttr<TYPE>(loop, LoopAttrKind::NAME, 0, 1);                      \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Loop &loop, TYPE val) {                           \
    ADD_1(LoopAttrKind, NAME, loop, val);                                      \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Loop &loop, raw_ostream *os) {           \
    VERIFY_1(os, loop, NAME, IRNAME, TYPE);                                    \
  }

#define LOOP_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)        \
  void llvm::add##NAME##Attr(Loop &loop, ETY0 e0, ETY1 e1) {                   \
    ADD_2(LoopAttrKind, NAME, loop, e0, e1);                                   \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Loop &loop, raw_ostream *os) {           \
    VERIFY_2(os, loop, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1);              \
  }

#define LOOP_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2)                                               \
  void llvm::add##NAME##Attr(Loop &loop, ETY0 e0, ETY1 e1, ETY2 e2) {          \
    ADD_3(LoopAttrKind, NAME, loop, e0, e1, e2);                               \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Loop &loop, raw_ostream *os) {           \
    VERIFY_3(os, loop, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2,         \
             ENAME2);                                                          \
  }

#define LOOP_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3)                            \
  void llvm::add##NAME##Attr(Loop &loop, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3) { \
    ADD_4(LoopAttrKind, NAME, loop, e0, e1, e2, e3);                           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Loop &loop, raw_ostream *os) {           \
    VERIFY_4(os, loop, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2, \
             ETY3, ENAME3);                                                    \
  }

#define LOOP_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4)         \
  void llvm::add##NAME##Attr(Loop &loop, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,   \
                             ETY4 e4) {                                        \
    ADD_5(LoopAttrKind, NAME, loop, e0, e1, e2, e3, e4);                       \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Loop &loop, raw_ostream *os) {           \
    VERIFY_5(os, loop, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2, \
             ETY3, ENAME3, ETY4, ENAME4);                                      \
  }

#define LOOP_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5)                                               \
  void llvm::add##NAME##Attr(Loop &loop, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,   \
                             ETY4 e4, ETY5 e5) {                               \
    ADD_6(LoopAttrKind, NAME, loop, e0, e1, e2, e3, e4, e5);                   \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Loop &loop, raw_ostream *os) {           \
    VERIFY_6(os, loop, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2, \
             ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5);                        \
  }

#define LOOP_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6)                            \
  void llvm::add##NAME##Attr(Loop &loop, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,   \
                             ETY4 e4, ETY5 e5, ETY6 e6) {                      \
    ADD_7(LoopAttrKind, NAME, loop, e0, e1, e2, e3, e4, e5, e6);               \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Loop &loop, raw_ostream *os) {           \
    VERIFY_7(os, loop, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2, \
             ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5, ETY6, ENAME6);          \
  }

#define LOOP_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7)         \
  void llvm::add##NAME##Attr(Loop &loop, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,   \
                             ETY4 e4, ETY5 e5, ETY6 e6, ETY7 e7) {             \
    ADD_8(LoopAttrKind, NAME, loop, e0, e1, e2, e3, e4, e5, e6, e7);           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Loop &loop, raw_ostream *os) {           \
    VERIFY_8(os, loop, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2, \
             ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5, ETY6, ENAME6, ETY7,     \
             ENAME7);                                                          \
  }

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

#define LOOP_ATTR_N(NAME, IRNAME, ETY, ENAME, EN, NELEMS)                      \
  std::optional<ETY> llvm::get##ENAME##From##NAME##Attr(const Loop &loop) {    \
    return getAttr<ETY>(loop, LoopAttrKind::NAME, EN, NELEMS);                 \
  }
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

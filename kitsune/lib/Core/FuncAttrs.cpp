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
#include "kitsune/Core/AttrsCommon.h"
#include "kitsune/Core/FuncUtils.h"
#include "kitsune/Core/MetadataUtils.h"
#include "kitsune/Support/Diagnostics.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Function.h"

using namespace llvm;

static void setAttrList(Function &f, MDNode *attrList) {
  f.setMetadata(LLVMContext::MD_kit_func_attrs, attrList);
}

static void addAttr(Function &f, StringRef name, ArrayRef<Metadata *> vals) {
  LLVMContext &ctx = f.getContext();
  MDNode *attrList = getAttrList(f);
  MDNode *newAttrList = getNewAttrListWith(name, vals, attrList, ctx);

  setAttrList(f, newAttrList);
}

static void removeAttr(Function &f, StringRef attrName) {
  MDNode *attrList = getAttrList(f);
  MDNode *newAttrList = getNewAttrListWithout(attrName, attrList);

  setAttrList(f, newAttrList);
}

MDNode *llvm::getAttrList(const Function &f) {
  return f.getMetadata(LLVMContext::MD_kit_func_attrs);
}

StringRef llvm::getAttrName(FuncAttrKind attr) {
  switch (attr) {
#define FUNC_ATTR(NAME, IRNAME, TYPE)                                          \
  case FuncAttrKind::NAME:                                                     \
    return IRNAME;
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
  }
  llvm_unreachable("getAttrName: Attribute not handled");
}

std::optional<FuncAttrKind> llvm::getFuncAttrKind(StringRef name) {
  return StringSwitch<std::optional<FuncAttrKind>>(name)
#define FUNC_ATTR(NAME, IRNAME, TYPE) .Case(IRNAME, FuncAttrKind::NAME)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
      .Default(std::nullopt);
}

bool llvm::verifyAttr(const Function &f, FuncAttrKind attr, raw_ostream *os) {
  switch (attr) {
#define FUNC_ATTR(NAME, IRNAME, TYPE)                                          \
  case FuncAttrKind::NAME:                                                     \
    return verify##NAME##Attr(f, os);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
  }
  llvm_unreachable("verifyAttr: Attribute not handled");
}

bool llvm::hasAttr(const Function &f, FuncAttrKind attr) {
  return getRawAttr(getAttrName(attr), getAttrList(f));
}

void llvm::addAttr(Function &f, FuncAttrKind attr) {
  StringRef attrName = getAttrName(attr);
  switch (attr) {
  default:
    emitDiagnostic(DiagID::ErrAttrWithoutValues, attrName);
    exitOnError();
    break;
#define FUNC_ATTR_0(NAME, IRNAME) case FuncAttrKind::NAME:
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
    return ::addAttr(f, attrName, {});
  }
}

void llvm::removeAttr(Function &f, FuncAttrKind attr) {
  ::removeAttr(f, getAttrName(attr));
}

#define FUNC_ATTR(NAME, IRNAME, TYPE)                                          \
  bool llvm::has##NAME##Attr(const Function &f) {                              \
    return getRawAttr(IRNAME, getAttrList(f));                                 \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(Function &f) { ::removeAttr(f, IRNAME); }

#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

#define FUNC_ATTR_LOOP(NAME, IRNAME)                                           \
  std::optional<Loop *> llvm::get##NAME##Attr(                                 \
      const Function &f, const SmallVectorImpl<const LoopInfo *> &lis) {       \
    return getAttrValue(IRNAME, getAttrList(f), lis);                          \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Function &f, const Loop &loop) {                  \
    ::addAttr(f, IRNAME, loop.getLoopID());                                    \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Function &f, raw_ostream *os) {          \
    return verifyAttrLoop(IRNAME, getAttrList(f), os);                         \
  }

#define FUNC_ATTR_0(NAME, IRNAME)                                              \
  void llvm::add##NAME##Attr(Function &f) { ADD_0(IRNAME, f); }                \
                                                                               \
  bool llvm::verify##NAME##Attr(const Function &f, raw_ostream *os) {          \
    return verifyAttr0(IRNAME, getAttrList(f), os);                            \
  }

#define FUNC_ATTR_1(NAME, IRNAME, TYPE)                                        \
  std::optional<TYPE> llvm::get##NAME##Attr(const Function &f) {               \
    return getAttrValue<TYPE>(IRNAME, getAttrList(f), 0, 1);                   \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Function &f, TYPE val) { ADD_1(IRNAME, f, val); } \
                                                                               \
  bool llvm::verify##NAME##Attr(const Function &f, raw_ostream *os) {          \
    VERIFY_1(os, f, NAME, IRNAME, TYPE);                                       \
  }

#define FUNC_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)        \
  void llvm::add##NAME##Attr(Function &f, ETY0 e0, ETY1 e1) {                  \
    ADD_2(IRNAME, f, e0, e1);                                                  \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Function &f, raw_ostream *os) {          \
    VERIFY_2(os, f, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1);                 \
  }

#define FUNC_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2)                                               \
  void llvm::add##NAME##Attr(Function &f, ETY0 e0, ETY1 e1, ETY2 e2) {         \
    ADD_3(IRNAME, f, e0, e1, e2);                                              \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Function &f, raw_ostream *os) {          \
    VERIFY_3(os, f, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2);   \
  }

#define FUNC_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3)                            \
  void llvm::add##NAME##Attr(Function &f, ETY0 e0, ETY1 e1, ETY2 e2,           \
                             ETY3 e3) {                                        \
    ADD_4(IRNAME, f, e0, e1, e2, e3);                                          \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Function &f, raw_ostream *os) {          \
    VERIFY_4(os, f, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2,    \
             ETY3, ENAME3);                                                    \
  }

#define FUNC_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4)         \
  void llvm::add##NAME##Attr(Function &f, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,  \
                             ETY4 e4) {                                        \
    ADD_5(IRNAME, f, e0, e1, e2, e3, e4);                                      \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Function &f, raw_ostream *os) {          \
    VERIFY_5(os, f, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2,    \
             ETY3, ENAME3, ETY4, ENAME4);                                      \
  }

#define FUNC_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5)                                               \
  void llvm::add##NAME##Attr(Function &f, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,  \
                             ETY4 e4, ETY5 e5) {                               \
    ADD_6(IRNAME, f, e0, e1, e2, e3, e4, e5);                                  \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Function &f, raw_ostream *os) {          \
    VERIFY_6(os, f, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2,    \
             ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5);                        \
  }

#define FUNC_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6)                            \
  void llvm::add##NAME##Attr(Function &f, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,  \
                             ETY4 e4, ETY5 e5, ETY6 e6) {                      \
    ADD_7(IRNAME, f, e0, e1, e2, e3, e4, e5, e6);                              \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Function &f, raw_ostream *os) {          \
    VERIFY_7(os, f, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2,    \
             ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5, ETY6, ENAME6);          \
  }

#define FUNC_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7)         \
  void llvm::add##NAME##Attr(Function &f, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,  \
                             ETY4 e4, ETY5 e5, ETY6 e6, ETY7 e7) {             \
    ADD_8(IRNAME, f, e0, e1, e2, e3, e4, e5, e6, e7);                          \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Function &f, raw_ostream *os) {          \
    VERIFY_8(os, f, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2,    \
             ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5, ETY6, ENAME6, ETY7,     \
             ENAME7);                                                          \
  }

#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

#define FUNC_ATTR_N(NAME, IRNAME, ETY, ENAME, EN, NELEMS)                      \
  std::optional<ETY> llvm::get##ENAME##From##NAME##Attr(const Function &f) {   \
    return getAttrValue<ETY>(IRNAME, getAttrList(f), EN, NELEMS);              \
  }
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

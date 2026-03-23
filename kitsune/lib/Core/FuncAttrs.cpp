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
#include "kitsune/Core/MetadataUtils.h"
#include "kitsune/Support/Diagnostics.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Function.h"

using namespace llvm;

static void addAttr(Function &f, FuncAttrKind attr, ArrayRef<Metadata *> ops) {
  LLVMContext &ctx = f.getContext();
  StringRef attrName = getAttrName(attr);
  MDNode *md = MDNode::get(ctx, ops);

  removeAttr(f, attr);
  f.addMetadata(attrName, *md);
}

template <typename T>
static std::optional<T> getAttr(const Function &f, FuncAttrKind attr,
                                unsigned i, unsigned n) {
  StringRef attrName = getAttrName(attr);
  if (MDNode *md = f.getMetadata(attrName))
    if (md->getNumOperands() == n)
      return fromMetadata<T>(md->getOperand(i));
  return std::nullopt;
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

bool llvm::hasAttr(const Function &f, FuncAttrKind attr) {
  return f.hasMetadata(getAttrName(attr));
}

void llvm::addAttr(Function &f, FuncAttrKind attr) {
  switch (attr) {
  default:
    emitDiagnostic(DiagID::ErrAttrWithoutValues, getAttrName(attr));
    exitOnError();
    break;
#define FUNC_ATTR_0(NAME, IRNAME) case FuncAttrKind::NAME:
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
    return ::addAttr(f, attr, {});
  }
}

void llvm::removeAttr(Function &f, FuncAttrKind attr) {
  f.setMetadata(getAttrName(attr), nullptr);
}

#define FUNC_ATTR(NAME, IRNAME, TYPE)                                          \
  bool llvm::has##NAME##Attr(const Function &f) {                              \
    return hasAttr(f, FuncAttrKind::NAME);                                     \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(Function &f) {                                 \
    removeAttr(f, FuncAttrKind::NAME);                                         \
  }
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

#define FUNC_ATTR_0(NAME, IRNAME)                                              \
  void llvm::add##NAME##Attr(Function &f) {                                    \
    ::addAttr(f, FuncAttrKind::NAME, {});                                      \
  }

#define FUNC_ATTR_1(NAME, IRNAME, TYPE)                                        \
  std::optional<TYPE> llvm::get##NAME##Attr(const Function &f) {               \
    return getAttr<TYPE>(f, FuncAttrKind::NAME, 0, 1);                         \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Function &f, TYPE val) {                          \
    Metadata *ops[] = {toMetadata(val, ctx)};                                  \
    :addAttr(f, FuncAttrKind::NAME, ops);                                      \
  }

#define FUNC_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)        \
  void llvm::add##NAME##Attr(Function &f, ETY0 e0, ETY1 e1) {                  \
    LLVMContext &ctx = f.getContext();                                         \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx)};              \
    ::addAttr(f, FuncAttrKind::NAME, ops);                                     \
  }

#define FUNC_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2)                                               \
  void llvm::add##NAME##Attr(Function &f, ETY0 e0, ETY1 e1, ETY2 e2) {         \
    LLVMContext &ctx = f.getContext();                                         \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx)};                                   \
    ::addAttr(f, FuncAttrKind::NAME, ops);                                     \
  }

#define FUNC_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3)                            \
  void llvm::add##NAME##Attr(Function &f, ETY0 e0, ETY1 e1, ETY2 e2,           \
                             ETY3 e3) {                                        \
    LLVMContext &ctx = f.getContext();                                         \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx), toMetadata(e3, ctx)};              \
    ::addAttr(f, FuncAttrKind::NAME, ops);                                     \
  }

#define FUNC_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4)         \
  void llvm::add##NAME##Attr(Function &f, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,  \
                             ETY4 e4) {                                        \
    LLVMContext &ctx = f.getContext();                                         \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx), toMetadata(e3, ctx),               \
                       toMetadata(e4, ctx)};                                   \
    ::addAttr(f, FuncAttrKind::NAME, ops);                                     \
  }

#define FUNC_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5)                                               \
  void llvm::add##NAME##Attr(Function &f, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,  \
                             ETY4 e4, ETY5 e5) {                               \
    LLVMContext &ctx = f.getContext();                                         \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx), toMetadata(e3, ctx),               \
                       toMetadata(e4, ctx), toMetadata(e5, ctx)};              \
    ::addAttr(f, FuncAttrKind::NAME, ops);                                     \
  }

#define FUNC_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6)                            \
  void llvm::add##NAME##Attr(Function &f, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,  \
                             ETY4 e4, ETY5 e5, ETY6 e6) {                      \
    LLVMContext &ctx = f.getContext();                                         \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx), toMetadata(e3, ctx),               \
                       toMetadata(e4, ctx), toMetadata(e5, ctx),               \
                       toMetadata(e6, ctx)};                                   \
    ::addAttr(f, FuncAttrKind::NAME, ops);                                     \
  }

#define FUNC_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7)         \
  void llvm::add##NAME##Attr(Function &f, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,  \
                             ETY4 e4, ETY5 e5, ETY6 e6, ETY7 e7) {             \
    LLVMContext &ctx = f.getContext();                                         \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx), toMetadata(e3, ctx),               \
                       toMetadata(e4, ctx), toMetadata(e5, ctx),               \
                       toMetadata(e6, ctx), toMetadata(e7, ctx)};              \
    ::addAttr(f, FuncAttrKind::NAME, ops);                                     \
  }
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

#define FUNC_ATTR_N(NAME, IRNAME, ETY, ENAME, EN, NELEMS)                      \
  std::optional<ETY> llvm::get##ENAME##From##NAME##Attr(const Function &f) {   \
    return getAttr<ETY>(f, FuncAttrKind::NAME, EN, NELEMS);                    \
  }
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

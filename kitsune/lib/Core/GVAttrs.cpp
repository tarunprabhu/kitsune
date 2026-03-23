//===- GVAttrs.cpp - Kitsune-specific attributes for global variables -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions and utilities to work with Kitsune-specific attributes for global
// variables.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/GVAttrs.h"
#include "kitsune/Core/MetadataUtils.h"
#include "kitsune/Support/Diagnostics.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/GlobalVariable.h"

using namespace llvm;

static void addAttr(GlobalVariable &g, GVAttrKind attr,
                    ArrayRef<Metadata *> ops) {
  LLVMContext &ctx = g.getContext();
  StringRef attrName = getAttrName(attr);
  MDNode *md = MDNode::get(ctx, ops);

  removeAttr(g, attr);
  g.addMetadata(attrName, *md);
}

template <typename T>
static std::optional<T> getAttr(const GlobalVariable &g, GVAttrKind attr,
                                unsigned i, unsigned n) {
  StringRef attrName = getAttrName(attr);
  if (MDNode *md = g.getMetadata(attrName))
    if (md->getNumOperands() == n)
      return fromMetadata<T>(md->getOperand(i));
  return std::nullopt;
}

StringRef llvm::getAttrName(GVAttrKind attr) {
  switch (attr) {
#define GV_ATTR(NAME, IRNAME, TYPE)                                            \
  case GVAttrKind::NAME:                                                       \
    return IRNAME;
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
  }
  llvm_unreachable("getAttrName: Attribute not handled");
}

std::optional<GVAttrKind> llvm::getGVAttrKind(StringRef name) {
  return StringSwitch<std::optional<GVAttrKind>>(name)
#define GV_ATTR(NAME, IRNAME, TYPE) .Case(IRNAME, GVAttrKind::NAME)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
      .Default(std::nullopt);
}

bool llvm::verifyAttr(const GlobalVariable &g, GVAttrKind attr) {
  switch (attr) {
#define GV_ATTR(NAME, IRNAME, TYPE)                                            \
  case GVAttrKind::NAME:                                                       \
    return verify##NAME##Attr(g);
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
  }
  llvm_unreachable("verifyAttr: Attribute not handled");
}

bool llvm::hasAttr(const GlobalVariable &g, GVAttrKind attr) {
  return g.hasMetadata(getAttrName(attr));
}

void llvm::addAttr(GlobalVariable &g, GVAttrKind attr) {
  switch (attr) {
  default:
    emitDiagnostic(DiagID::ErrAttrWithoutValues, getAttrName(attr));
    exitOnError();
    break;
#define GV_ATTR_0(NAME, IRNAME) case GVAttrKind::NAME:
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
    return ::addAttr(g, attr, {});
  }
}

void llvm::removeAttr(GlobalVariable &g, GVAttrKind attr) {
  g.setMetadata(getAttrName(attr), nullptr);
}

#define GV_ATTR(NAME, IRNAME, TYPE)                                            \
  bool llvm::has##NAME##Attr(const GlobalVariable &g) {                        \
    return hasAttr(g, GVAttrKind::NAME);                                       \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(GlobalVariable &g) {                           \
    removeAttr(g, GVAttrKind::NAME);                                           \
  }

#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

#define GV_ATTR_0(NAME, IRNAME)                                                \
  void llvm::add##NAME##Attr(GlobalVariable &g) {                              \
    ::addAttr(g, GVAttrKind::NAME, {});                                        \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g) {                     \
    if (MDNode *md = g.getMetadata(IRNAME))                                    \
      return md->getNumOperands() == 0;                                        \
    return true;                                                               \
  }

#define GV_ATTR_1(NAME, IRNAME, TYPE)                                          \
  std::optional<TYPE> llvm::get##NAME##Attr(const GlobalVariable &g) {         \
    return getAttr<TYPE>(g, GVAttrKind::NAME, 0, 1);                           \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(GlobalVariable &g, TYPE val) {                    \
    LLVMContext &ctx = g.getContext();                                         \
    Metadata *ops[] = {toMetadata(val, ctx)};                                  \
    ::addAttr(g, GVAttrKind::NAME, ops);                                       \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g) {                     \
    if (has##NAME##Attr(g))                                                    \
      return get##NAME##Attr(g).has_value();                                   \
    return true;                                                               \
  }

#define GV_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)          \
  void llvm::add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1) {            \
    LLVMContext &ctx = g.getContext();                                         \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx)};              \
    ::addAttr(g, GVAttrKind::NAME, ops);                                       \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g) {                     \
    if (has##NAME##Attr(g))                                                    \
      return get##ENAME0##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME1##From##NAME##Attr(g).has_value();                     \
    return true;                                                               \
  }

#define GV_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2)                                                 \
  void llvm::add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1, ETY2 e2) {   \
    LLVMContext &ctx = g.getContext();                                         \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx)};                                   \
    ::addAttr(g, GVAttrKind::NAME, ops);                                       \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g) {                     \
    if (has##NAME##Attr(g))                                                    \
      return get##ENAME0##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME1##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME2##From##NAME##Attr(g).has_value();                     \
    return true;                                                               \
  }

#define GV_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2, ETY3, ENAME3, EN3)                              \
  void llvm::add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1, ETY2 e2,     \
                             ETY3 e3) {                                        \
    LLVMContext &ctx = g.getContext();                                         \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx), toMetadata(e3, ctx)};              \
    ::addAttr(g, GVAttrKind::NAME, ops);                                       \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g) {                     \
    if (has##NAME##Attr(g))                                                    \
      return get##ENAME0##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME1##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME2##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME3##From##NAME##Attr(g).has_value();                     \
    return true;                                                               \
  }

#define GV_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4)           \
  void llvm::add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1, ETY2 e2,     \
                             ETY3 e3, ETY4 e4) {                               \
    LLVMContext &ctx = g.getContext();                                         \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx), toMetadata(e3, ctx),               \
                       toMetadata(e4, ctx)};                                   \
    ::addAttr(g, GVAttrKind::NAME, ops);                                       \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g) {                     \
    if (has##NAME##Attr(g))                                                    \
      return get##ENAME0##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME1##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME2##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME3##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME4##From##NAME##Attr(g).has_value();                     \
    return true;                                                               \
  }

#define GV_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,     \
                  ENAME5, EN5)                                                 \
  void llvm::add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1, ETY2 e2,     \
                             ETY3 e3, ETY4 e4, ETY5 e5) {                      \
    LLVMContext &ctx = g.getContext();                                         \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx), toMetadata(e3, ctx),               \
                       toMetadata(e4, ctx), toMetadata(e5, ctx)};              \
    ::addAttr(g, GVAttrKind::NAME, ops);                                       \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g) {                     \
    if (has##NAME##Attr(g))                                                    \
      return get##ENAME0##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME1##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME2##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME3##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME4##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME5##From##NAME##Attr(g).has_value();                     \
    return true;                                                               \
  }

#define GV_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,     \
                  ENAME5, EN5, ETY6, ENAME6, EN6)                              \
  void llvm::add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1, ETY2 e2,     \
                             ETY3 e3, ETY4 e4, ETY5 e5, ETY6 e6) {             \
    LLVMContext &ctx = g.getContext();                                         \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx), toMetadata(e3, ctx),               \
                       toMetadata(e4, ctx), toMetadata(e5, ctx),               \
                       toMetadata(e6, ctx)};                                   \
    ::addAttr(g, GVAttrKind::NAME, ops);                                       \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g) {                     \
    if (has##NAME##Attr(g))                                                    \
      return get##ENAME0##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME1##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME2##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME3##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME4##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME5##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME6##From##NAME##Attr(g).has_value();                     \
    return true;                                                               \
  }

#define GV_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,     \
                  ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7)           \
  void llvm::add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1, ETY2 e2,     \
                             ETY3 e3, ETY4 e4, ETY5 e5, ETY6 e6, ETY7 e7) {    \
    LLVMContext &ctx = g.getContext();                                         \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx), toMetadata(e3, ctx),               \
                       toMetadata(e4, ctx), toMetadata(e5, ctx),               \
                       toMetadata(e6, ctx), toMetadata(e7, ctx)};              \
    ::addAttr(g, GVAttrKind::NAME, ops);                                       \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const GlobalVariable &g) {                     \
    if (has##NAME##Attr(g))                                                    \
      return get##ENAME0##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME1##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME2##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME3##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME4##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME5##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME6##From##NAME##Attr(g).has_value() &&                   \
             get##ENAME7##From##NAME##Attr(g).has_value();                     \
    return true;                                                               \
  }

#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

#define GV_ATTR_N(NAME, IRNAME, ETY, ENAME, EN, NELEMS)                        \
  std::optional<ETY> llvm::get##ENAME##From##NAME##Attr(                       \
      const GlobalVariable &g) {                                               \
    return getAttr<ETY>(g, GVAttrKind::NAME, EN, NELEMS);                      \
  }
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

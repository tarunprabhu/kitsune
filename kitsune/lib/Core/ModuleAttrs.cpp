//===- ModuleAttrs.cpp - Module attributes and utilities ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions and utilities to work with module "attributes" (really named
// LLVM-IR metadata).
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ModuleAttrs.h"
#include "kitsune/Core/MetadataUtils.h"
#include "kitsune/Support/Diagnostics.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Module.h"

using namespace llvm;

static void addAttr(Module &m, ModuleAttrKind attr, ArrayRef<Metadata *> mds) {
  removeAttr(m, attr);

  LLVMContext &ctx = m.getContext();
  StringRef attrName = getAttrName(attr);
  NamedMDNode *nmd = m.getOrInsertNamedMetadata(attrName);
  for (Metadata *md : mds)
    nmd->addOperand(MDNode::get(ctx, md));
}

template <typename T>
static std::optional<T> getAttr(const Module &m, ModuleAttrKind attr,
                                unsigned i, unsigned n) {
  StringRef attrName = getAttrName(attr);
  if (NamedMDNode *nmd = m.getNamedMetadata(attrName)) {
    if (nmd->getNumOperands() == n) {
      MDNode *md = nmd->getOperand(i);
      if (md->getNumOperands() == 1)
        return fromMetadata<T>(md->getOperand(0));
    }
  }
  return std::nullopt;
}

StringRef llvm::getAttrName(ModuleAttrKind attrKind) {
  switch (attrKind) {
#define MODULE_ATTR(NAME, IRNAME, TYPE)                                        \
  case ModuleAttrKind::NAME:                                                   \
    return IRNAME;
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
  }
  llvm_unreachable("getAttrName: ModuleAttrKind not handled");
}

std::optional<ModuleAttrKind> llvm::getModuleAttrKind(StringRef name) {
  return StringSwitch<std::optional<ModuleAttrKind>>(name)
#define MODULE_ATTR(NAME, IRNAME, TYPE) .Case(IRNAME, ModuleAttrKind::NAME)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
      .Default(std::nullopt);
}

bool llvm::verifyAttr(const Module &m, ModuleAttrKind attr) {
  switch (attr) {
#define MODULE_ATTR(NAME, IRNAME, TYPE)                                        \
  case ModuleAttrKind::NAME:                                                   \
    return verify##NAME##Attr(m);
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
  }
  llvm_unreachable("verifyAttr: Attribute not handled");
}

bool llvm::hasAttr(const Module &m, ModuleAttrKind attr) {
  return m.getNamedMetadata(getAttrName(attr));
}

void llvm::addAttr(Module &m, ModuleAttrKind attr) {
  switch (attr) {
  default:
    emitDiagnostic(DiagID::ErrAttrWithoutValues, getAttrName(attr));
    exitOnError();
    break;
#define MODULE_ATTR_0(NAME, IRNAME) case ModuleAttrKind::NAME:
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
    return ::addAttr(m, attr, {});
  }
}

void llvm::removeAttr(Module &m, ModuleAttrKind attr) {
  StringRef attrName = getAttrName(attr);
  if (NamedMDNode *nmd = m.getNamedMetadata(attrName))
    m.eraseNamedMetadata(nmd);
}

#define MODULE_ATTR(NAME, IRNAME, TYPE)                                        \
  bool llvm::has##NAME##Attr(const Module &m) {                                \
    return hasAttr(m, ModuleAttrKind::NAME);                                   \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(Module &m) {                                   \
    removeAttr(m, ModuleAttrKind::NAME);                                       \
  }
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

#define MODULE_ATTR_0(NAME, IRNAME)                                            \
  void llvm::add##NAME##Attr(Module &m) {                                      \
    ::addAttr(m, ModuleAttrKind::NAME, {});                                    \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Module &m) {                             \
    if (NamedMDNode *nmd = m.getNamedMetadata(IRNAME))                         \
      return nmd->getNumOperands() == 0;                                       \
    return true;                                                               \
  }

#define MODULE_ATTR_1(NAME, IRNAME, TYPE)                                      \
  std::optional<TYPE> llvm::get##NAME##Attr(const Module &m) {                 \
    return getAttr<TYPE>(m, ModuleAttrKind::NAME, 0, 1);                       \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Module &m, TYPE val) {                            \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *ops[] = {toMetadata(val, ctx)};                                  \
    ::addAttr(m, ModuleAttrKind::NAME, ops);                                   \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Module &m) {                             \
    if (has##NAME##Attr(m))                                                    \
      return get##NAME##Attr(m).has_value();                                   \
    return true;                                                               \
  }

#define MODULE_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)      \
  void llvm::add##NAME##Attr(Module &m, ETY0 e0, ETY1 e1) {                    \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx)};              \
    ::addAttr(m, ModuleAttrKind::NAME, ops);                                   \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Module &m) {                             \
    if (has##NAME##Attr(m))                                                    \
      return get##ENAME0##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME1##From##NAME##Attr(m).has_value();                     \
    return true;                                                               \
  }

#define MODULE_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2)                                       \
  void llvm::add##NAME##Attr(Module &m, ETY0 e0, ETY1 e1, ETY2 e2) {           \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx)};                                   \
    ::addAttr(m, ModuleAttrKind::NAME, ops);                                   \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Module &m) {                             \
    if (has##NAME##Attr(m))                                                    \
      return get##ENAME0##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME1##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME2##From##NAME##Attr(m).has_value();                     \
    return true;                                                               \
  }

#define MODULE_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3)                    \
  void llvm::add##NAME##Attr(Module &m, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3) {  \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx), toMetadata(e3, ctx)};              \
    ::addAttr(m, ModuleAttrKind::NAME, ops);                                   \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Module &m) {                             \
    if (has##NAME##Attr(m))                                                    \
      return get##ENAME0##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME1##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME2##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME3##From##NAME##Attr(m).has_value();                     \
    return true;                                                               \
  }

#define MODULE_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4) \
  void llvm::add##NAME##Attr(Module &m, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,    \
                             ETY4 e4) {                                        \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx), toMetadata(e3, ctx),               \
                       toMetadata(e4, ctx)};                                   \
    ::addAttr(m, ModuleAttrKind::NAME, ops);                                   \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Module &m) {                             \
    if (has##NAME##Attr(m))                                                    \
      return get##ENAME0##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME1##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME2##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME3##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME4##From##NAME##Attr(m).has_value();                     \
    return true;                                                               \
  }

#define MODULE_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, \
                      ETY5, ENAME5, EN5)                                       \
  void llvm::add##NAME##Attr(Module &m, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,    \
                             ETY4 e4, ETY5 e5) {                               \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx), toMetadata(e3, ctx),               \
                       toMetadata(e4, ctx), toMetadata(e5, ctx)};              \
    ::addAttr(m, ModuleAttrKind::NAME, ops);                                   \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Module &m) {                             \
    if (has##NAME##Attr(m))                                                    \
      return get##ENAME0##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME1##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME2##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME3##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME4##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME5##From##NAME##Attr(m).has_value();                     \
    return true;                                                               \
  }

#define MODULE_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, \
                      ETY5, ENAME5, EN5, ETY6, ENAME6, EN6)                    \
  void llvm::add##NAME##Attr(Module &m, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,    \
                             ETY4 e4, ETY5 e5, ETY6 e6) {                      \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx), toMetadata(e3, ctx),               \
                       toMetadata(e4, ctx), toMetadata(e5, ctx),               \
                       toMetadata(e6, ctx)};                                   \
    ::addAttr(m, ModuleAttrKind::NAME, ops);                                   \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Module &m) {                             \
    if (has##NAME##Attr(m))                                                    \
      return get##ENAME0##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME1##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME2##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME3##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME4##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME5##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME6##From##NAME##Attr(m).has_value();                     \
    return true;                                                               \
  }

#define MODULE_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, \
                      ETY5, ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7) \
  void llvm::add##NAME##Attr(Module &m, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,    \
                             ETY4 e4, ETY5 e5, ETY6 e6, ETY7 e7) {             \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *ops[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),               \
                       toMetadata(e2, ctx), toMetadata(e3, ctx),               \
                       toMetadata(e4, ctx), toMetadata(e5, ctx),               \
                       toMetadata(e6, ctx), toMetadata(e7, ctx)};              \
    ::addAttr(m, ModuleAttrKind::NAME, ops);                                   \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Module &m) {                             \
    if (has##NAME##Attr(m))                                                    \
      return get##ENAME0##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME1##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME2##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME3##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME4##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME5##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME6##From##NAME##Attr(m).has_value() &&                   \
             get##ENAME7##From##NAME##Attr(m).has_value();                     \
    return true;                                                               \
  }

#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

#define MODULE_ATTR_N(NAME, IRNAME, ETY, ENAME, EN, NELEMS)                    \
  std::optional<ETY> llvm::get##ENAME##From##NAME##Attr(const Module &m) {     \
    return getAttr<ETY>(m, ModuleAttrKind::NAME, EN, NELEMS);                  \
  }
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

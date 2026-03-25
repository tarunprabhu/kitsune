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
#include "kitsune/Core/AttrsCommon.h"
#include "kitsune/Core/MetadataUtils.h"
#include "kitsune/Core/ModuleUtils.h"
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

bool llvm::verifyAttr(const Module &m, ModuleAttrKind attr, raw_ostream *os) {
  switch (attr) {
#define MODULE_ATTR(NAME, IRNAME, TYPE)                                        \
  case ModuleAttrKind::NAME:                                                   \
    return verify##NAME##Attr(m, os);
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
  void llvm::add##NAME##Attr(Module &m) { ADD_0(ModuleAttrKind, NAME, m); }    \
                                                                               \
  bool llvm::verify##NAME##Attr(const Module &m, raw_ostream *os) {            \
    if (NamedMDNode *nmd = m.getNamedMetadata(IRNAME))                         \
      VERIFY_0(nmd->getNumOperands() == 0, IRNAME, os);                        \
    return true;                                                               \
  }

#define MODULE_ATTR_1(NAME, IRNAME, TYPE)                                      \
  std::optional<TYPE> llvm::get##NAME##Attr(const Module &m) {                 \
    return getAttr<TYPE>(m, ModuleAttrKind::NAME, 0, 1);                       \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Module &m, TYPE val) {                            \
    ADD_1(ModuleAttrKind, NAME, m, val);                                       \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Module &m, raw_ostream *os) {            \
    VERIFY_1(os, m, NAME, IRNAME, TYPE);                                       \
  }

#define MODULE_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)      \
  void llvm::add##NAME##Attr(Module &m, ETY0 e0, ETY1 e1) {                    \
    ADD_2(ModuleAttrKind, NAME, m, e0, e1);                                    \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Module &m, raw_ostream *os) {            \
    VERIFY_2(os, m, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1);                 \
  }

#define MODULE_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2)                                       \
  void llvm::add##NAME##Attr(Module &m, ETY0 e0, ETY1 e1, ETY2 e2) {           \
    ADD_3(ModuleAttrKind, NAME, m, e0, e1, e2);                                \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Module &m, raw_ostream *os) {            \
    VERIFY_3(os, m, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2);   \
  }

#define MODULE_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3)                    \
  void llvm::add##NAME##Attr(Module &m, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3) {  \
    ADD_4(ModuleAttrKind, NAME, m, e0, e1, e2, e3);                            \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Module &m, raw_ostream *os) {            \
    VERIFY_4(os, m, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2,    \
             ETY3, ENAME3);                                                    \
  }

#define MODULE_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4) \
  void llvm::add##NAME##Attr(Module &m, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,    \
                             ETY4 e4) {                                        \
    ADD_5(ModuleAttrKind, NAME, m, e0, e1, e2, e3, e4);                        \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Module &m, raw_ostream *os) {            \
    VERIFY_5(os, m, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2,    \
             ETY3, ENAME3, ETY4, ENAME4);                                      \
  }

#define MODULE_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, \
                      ETY5, ENAME5, EN5)                                       \
  void llvm::add##NAME##Attr(Module &m, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,    \
                             ETY4 e4, ETY5 e5) {                               \
    ADD_6(ModuleAttrKind, NAME, m, e0, e1, e2, e3, e4, e5);                    \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Module &m, raw_ostream *os) {            \
    VERIFY_6(os, m, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2,    \
             ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5);                        \
  }

#define MODULE_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, \
                      ETY5, ENAME5, EN5, ETY6, ENAME6, EN6)                    \
  void llvm::add##NAME##Attr(Module &m, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,    \
                             ETY4 e4, ETY5 e5, ETY6 e6) {                      \
    ADD_7(ModuleAttrKind, NAME, m, e0, e1, e2, e3, e4, e5, e6);                \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Module &m, raw_ostream *os) {            \
    VERIFY_7(os, m, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2,    \
             ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5, ETY6, ENAME6);          \
  }

#define MODULE_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, \
                      ETY5, ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7) \
  void llvm::add##NAME##Attr(Module &m, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,    \
                             ETY4 e4, ETY5 e5, ETY6 e6, ETY7 e7) {             \
    ADD_8(ModuleAttrKind, NAME, m, e0, e1, e2, e3, e4, e5, e6, e7);            \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const Module &m, raw_ostream *os) {            \
    VERIFY_8(os, m, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2, ENAME2,    \
             ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5, ETY6, ENAME6, ETY7,     \
             ENAME7);                                                          \
  }

#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

#define MODULE_ATTR_N(NAME, IRNAME, ETY, ENAME, EN, NELEMS)                    \
  std::optional<ETY> llvm::get##ENAME##From##NAME##Attr(const Module &m) {     \
    return getAttr<ETY>(m, ModuleAttrKind::NAME, EN, NELEMS);                  \
  }
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

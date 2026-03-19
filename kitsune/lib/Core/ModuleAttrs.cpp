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

static void addAttr(Module &m, ModuleAttrKind kind, ArrayRef<Metadata *> mds) {
  removeAttr(m, kind);

  LLVMContext &ctx = m.getContext();
  NamedMDNode *nmd = m.getOrInsertNamedMetadata(getAttrName(kind));
  for (Metadata *md : mds)
    nmd->addOperand(MDNode::get(ctx, md));
}

StringRef llvm::getAttrName(ModuleAttrKind attrKind) {
  switch (attrKind) {
#define MODULE_ATTR(NAME, IRNAME)                                              \
  case ModuleAttrKind::NAME:                                                   \
    return IRNAME;
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
  }
  llvm_unreachable("getAttrName: ModuleAttrKind not handled");
}

std::optional<ModuleAttrKind> llvm::getModuleAttrKind(StringRef name) {
  return StringSwitch<std::optional<ModuleAttrKind>>(name)
#define MODULE_ATTR(NAME, IRNAME) .Case(IRNAME, ModuleAttrKind::NAME)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
      .Default(std::nullopt);
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
  if (NamedMDNode *md = m.getNamedMetadata(getAttrName(attr)))
    m.eraseNamedMetadata(md);
}

#define MODULE_ATTR(NAME, IRNAME)                                              \
  bool llvm::has##NAME##Attr(const Module &m) {                                \
    return hasAttr(m, ModuleAttrKind::NAME);                                   \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(Module &m) {                                   \
    removeAttr(m, ModuleAttrKind::NAME);                                       \
  }

#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

#define MODULE_GETTER(NAME, TY, V, OP)                                         \
  std::optional<TY> llvm::get##V##From##NAME##Attr(const Module &m) {          \
    StringRef attrName = getAttrName(ModuleAttrKind::NAME);                    \
    if (NamedMDNode *nmd = m.getNamedMetadata(attrName))                       \
      if (MDNode *md = nmd->getOperand(OP))                                    \
        if (md->getNumOperands() > 0)                                          \
          return fromMetadata<TY>(md->getOperand(0));                          \
    return std::nullopt;                                                       \
  }

#define MODULE_ATTR_0(NAME, IRNAME)                                            \
  void llvm::add##NAME##Attr(Module &m) {                                      \
    return ::addAttr(m, ModuleAttrKind::NAME, {});                             \
  }

#define MODULE_ATTR_1(NAME, IRNAME, TY1, V1)                                   \
  void llvm::add##NAME##Attr(Module &m, TY1 V1) {                              \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *op0 = toMetadata(V1, ctx);                                       \
                                                                               \
    ::addAttr(m, ModuleAttrKind::NAME, {op0});                                 \
  }                                                                            \
  MODULE_GETTER(NAME, TY1, V1, 0)

#define MODULE_ATTR_2(NAME, IRNAME, TY1, V1, TY2, V2)                          \
  void llvm::add##NAME##Attr(Module &m, TY1 V1, TY2 V2) {                      \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *op0 = toMetadata(V1, ctx);                                       \
    Metadata *op1 = toMetadata(V2, ctx);                                       \
                                                                               \
    ::addAttr(m, ModuleAttrKind::NAME, {op0, op1});                            \
  }                                                                            \
  MODULE_GETTER(NAME, TY1, V1, 0)                                              \
  MODULE_GETTER(NAME, TY2, V2, 1)

#define MODULE_ATTR_3(NAME, IRNAME, TY1, V1, TY2, V2, TY3, V3)                 \
  void llvm::add##NAME##Attr(Module &m, TY1 V1, TY2 V2, TY3 V3) {              \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *op0 = toMetadata(V1, ctx);                                       \
    Metadata *op1 = toMetadata(V2, ctx);                                       \
    Metadata *op2 = toMetadata(V3, ctx);                                       \
                                                                               \
    ::addAttr(m, ModuleAttrKind::NAME, {op0, op1, op2});                       \
  }                                                                            \
  MODULE_GETTER(NAME, TY1, V1, 0)                                              \
  MODULE_GETTER(NAME, TY2, V2, 1)                                              \
  MODULE_GETTER(NAME, TY3, V3, 2)

#define MODULE_ATTR_4(NAME, IRNAME, TY1, V1, TY2, V2, TY3, V3, TY4, V4)        \
  void llvm::add##NAME##Attr(Module &m, TY1 V1, TY2 V2, TY3 V3, TY4 V4) {      \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *op0 = toMetadata(V1, ctx);                                       \
    Metadata *op1 = toMetadata(V2, ctx);                                       \
    Metadata *op2 = toMetadata(V3, ctx);                                       \
    Metadata *op3 = toMetadata(V4, ctx);                                       \
                                                                               \
    ::addAttr(m, ModuleAttrKind::NAME, {op0, op1, op2, op3});                  \
  }                                                                            \
  MODULE_GETTER(NAME, TY1, V1, 0)                                              \
  MODULE_GETTER(NAME, TY2, V2, 1)                                              \
  MODULE_GETTER(NAME, TY3, V3, 2)                                              \
  MODULE_GETTER(NAME, TY4, V4, 3)

#define MODULE_ATTR_5(NAME, IRNAME, TY1, V1, TY2, V2, TY3, V3, TY4, V4, TY5,   \
                      V5)                                                      \
  void llvm::add##NAME##Attr(Module &m, TY1 V1, TY2 V2, TY3 V3, TY4 V4,        \
                             TY5 V5) {                                         \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *op0 = toMetadata(V1, ctx);                                       \
    Metadata *op1 = toMetadata(V2, ctx);                                       \
    Metadata *op2 = toMetadata(V3, ctx);                                       \
    Metadata *op3 = toMetadata(V4, ctx);                                       \
    Metadata *op4 = toMetadata(V5, ctx);                                       \
                                                                               \
    ::addAttr(m, ModuleAttrKind::NAME, {op0, op1, op2, op3, op4});             \
  }                                                                            \
  MODULE_GETTER(NAME, TY1, V1, 0)                                              \
  MODULE_GETTER(NAME, TY2, V2, 1)                                              \
  MODULE_GETTER(NAME, TY3, V3, 2)                                              \
  MODULE_GETTER(NAME, TY4, V4, 3)                                              \
  MODULE_GETTER(NAME, TY5, V5, 4)

#define MODULE_ATTR_6(NAME, IRNAME, TY1, V1, TY2, V2, TY3, V3, TY4, V4, TY5,   \
                      V5, TY6, V6)                                             \
  void llvm::add##NAME##Attr(Module &m, TY1 V1, TY2 V2, TY3 V3, TY4 V4,        \
                             TY5 V5, TY6 V6) {                                 \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *op0 = toMetadata(V1, ctx);                                       \
    Metadata *op1 = toMetadata(V2, ctx);                                       \
    Metadata *op2 = toMetadata(V3, ctx);                                       \
    Metadata *op3 = toMetadata(V4, ctx);                                       \
    Metadata *op4 = toMetadata(V5, ctx);                                       \
    Metadata *op5 = toMetadata(V6, ctx);                                       \
                                                                               \
    ::addAttr(m, ModuleAttrKind::NAME, {op0, op1, op2, op3, op4, op5});        \
  }                                                                            \
  MODULE_GETTER(NAME, TY1, V1, 0)                                              \
  MODULE_GETTER(NAME, TY2, V2, 1)                                              \
  MODULE_GETTER(NAME, TY3, V3, 2)                                              \
  MODULE_GETTER(NAME, TY4, V4, 3)                                              \
  MODULE_GETTER(NAME, TY5, V5, 4)                                              \
  MODULE_GETTER(NAME, TY6, V6, 5)

#define MODULE_ATTR_7(NAME, IRNAME, TY1, V1, TY2, V2, TY3, V3, TY4, V4, TY5,   \
                      V5, TY6, V6, TY7, V7)                                    \
  void llvm::add##NAME##Attr(Module &m, TY1 V1, TY2 V2, TY3 V3, TY4 V4,        \
                             TY5 V5, TY6 V6, TY7 V7) {                         \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *op0 = toMetadata(V1, ctx);                                       \
    Metadata *op1 = toMetadata(V2, ctx);                                       \
    Metadata *op2 = toMetadata(V3, ctx);                                       \
    Metadata *op3 = toMetadata(V4, ctx);                                       \
    Metadata *op4 = toMetadata(V5, ctx);                                       \
    Metadata *op5 = toMetadata(V6, ctx);                                       \
    Metadata *op6 = toMetadata(V7, ctx);                                       \
                                                                               \
    ::addAttr(m, ModuleAttrKind::NAME, {op0, op1, op2, op3, op4, op5, op6});   \
  }                                                                            \
  MODULE_GETTER(NAME, TY1, V1, 0)                                              \
  MODULE_GETTER(NAME, TY2, V2, 1)                                              \
  MODULE_GETTER(NAME, TY3, V3, 2)                                              \
  MODULE_GETTER(NAME, TY4, V4, 3)                                              \
  MODULE_GETTER(NAME, TY5, V5, 4)                                              \
  MODULE_GETTER(NAME, TY6, V6, 5)                                              \
  MODULE_GETTER(NAME, TY7, V7, 6)

#define MODULE_ATTR_8(NAME, IRNAME, TY1, V1, TY2, V2, TY3, V3, TY4, V4, TY5,   \
                      V5, TY6, V6, TY7, V7, TY8, V8)                           \
  void llvm::add##NAME##Attr(Module &m, TY1 V1, TY2 V2, TY3 V3, TY4 V4,        \
                             TY5 V5, TY6 V6, TY7 V7, TY8 V8) {                 \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *op0 = toMetadata(V1, ctx);                                       \
    Metadata *op1 = toMetadata(V2, ctx);                                       \
    Metadata *op2 = toMetadata(V3, ctx);                                       \
    Metadata *op3 = toMetadata(V4, ctx);                                       \
    Metadata *op4 = toMetadata(V5, ctx);                                       \
    Metadata *op5 = toMetadata(V6, ctx);                                       \
    Metadata *op6 = toMetadata(V7, ctx);                                       \
    Metadata *op7 = toMetadata(V8, ctx);                                       \
                                                                               \
    ::addAttr(m, ModuleAttrKind::NAME,                                         \
              {op0, op1, op2, op3, op4, op5, op6, op7});                       \
  }                                                                            \
  MODULE_GETTER(NAME, TY1, V1, 0)                                              \
  MODULE_GETTER(NAME, TY2, V2, 1)                                              \
  MODULE_GETTER(NAME, TY3, V3, 2)                                              \
  MODULE_GETTER(NAME, TY4, V4, 3)                                              \
  MODULE_GETTER(NAME, TY5, V5, 4)                                              \
  MODULE_GETTER(NAME, TY6, V6, 5)                                              \
  MODULE_GETTER(NAME, TY7, V7, 6)                                              \
  MODULE_GETTER(NAME, TY8, V8, 7)

#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

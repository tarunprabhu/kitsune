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
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Module.h"

using namespace llvm;

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

/// Get the kind of the module attribute from the name that would appear in
/// metadata, otherwise, return std::nullopt.
std::optional<ModuleAttrKind> llvm::getModuleAttrKind(StringRef name) {
  return StringSwitch<std::optional<ModuleAttrKind>>(name)
#define MODULE_ATTR(NAME, IRNAME) .Case(IRNAME, ModuleAttrKind::NAME)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
      .Default(std::nullopt);
}

/// Check if the given attribute is present in a module.
bool llvm::hasAttr(const Module &m, ModuleAttrKind attr) {
  return m.getNamedMetadata(getAttrName(attr));
}

/// Remove the attribute from a module. If the loop does not contain the
/// attribute, this has no effect.
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

static NamedMDNode &addAttr(Module &m, ModuleAttrKind kind,
                            ArrayRef<Metadata *> mds) {
  removeAttr(m, kind);

  LLVMContext &ctx = m.getContext();
  NamedMDNode *nmd = m.getOrInsertNamedMetadata(getAttrName(kind));
  for (Metadata *md : mds)
    nmd->addOperand(MDNode::get(ctx, md));
  return *nmd;
}

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
  NamedMDNode &llvm::add##NAME##Attr(Module &m) {                              \
    return addAttr(m, ModuleAttrKind::NAME, {});                               \
  }

#define MODULE_ATTR_1(NAME, IRNAME, TY1, V1)                                   \
  NamedMDNode &lvm::add##NAME##Attr(Module &m, TY1 V1) {                       \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *op0 = toMetadata(ctx, V1);                                       \
    return addAttr(m, ModuleAttrKind::NAME, {op0});                            \
  }                                                                            \
  MODULE_GETTER(NAME, TY1, V1, 0)

#define MODULE_ATTR_2(NAME, IRNAME, TY1, V1, TY2, V2)                          \
  NamedMDNode &llvm::add##NAME##Attr(Module &m, TY1 V1, TY2 V2) {              \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *op0 = toMetadata(ctx, V1);                                       \
    Metadata *op1 = toMetadata(ctx, V2);                                       \
    return addAttr(m, ModuleAttrKind::NAME, {op0, op1});                       \
  }                                                                            \
  MODULE_GETTER(NAME, TY1, V1, 0)                                              \
  MODULE_GETTER(NAME, TY2, V2, 1)

#define MODULE_ATTR_3(NAME, IRNAME, TY1, V1, TY2, V2, TY3, V3)                 \
  NamedMDNode &llvm::add##NAME##Attr(Module &m, TY1 V1, TY2 V2, TY3 V3) {      \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *op0 = toMetadata(ctx, V1);                                       \
    Metadata *op1 = toMetadata(ctx, V2);                                       \
    Metadata *op2 = toMetadata(ctx, V3);                                       \
    return addAttr(m, ModuleAttrKind::NAME, {op0, op1, op2});                  \
  }                                                                            \
  MODULE_GETTER(NAME, TY1, V1, 0)                                              \
  MODULE_GETTER(NAME, TY2, V2, 1)                                              \
  MODULE_GETTER(NAME, TY3, V3, 2)

#define MODULE_ATTR_4(NAME, IRNAME, TY1, V1, TY2, V2, TY3, V3, TY4, V4)        \
  NamedMDNode &llvm::add##NAME##Attr(Module &m, TY1 V1, TY2 V2, TY3 V3,        \
                                     TY4 V4) {                                 \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *op0 = toMetadata(ctx, V1);                                       \
    Metadata *op1 = toMetadata(ctx, V2);                                       \
    Metadata *op2 = toMetadata(ctx, V3);                                       \
    Metadata *op3 = toMetadata(ctx, V4);                                       \
    return addAttr(m, ModuleAttrKind::NAME, {op0, op1, op2, op3});             \
  }                                                                            \
  MODULE_GETTER(NAME, TY1, V1, 0)                                              \
  MODULE_GETTER(NAME, TY2, V2, 1)                                              \
  MODULE_GETTER(NAME, TY3, V3, 2)                                              \
  MODULE_GETTER(NAME, TY4, V4, 3)

#define MODULE_ATTR_5(NAME, IRNAME, TY1, V1, TY2, V2, TY3, V3, TY4, V4, TY5,   \
                      V5)                                                      \
  NamedMDNode &llvm::add##NAME##Attr(Module &m, TY1 V1, TY2 V2, TY3 V3,        \
                                     TY4 V4, TY5 V5) {                         \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *op0 = toMetadata(ctx, V1);                                       \
    Metadata *op1 = toMetadata(ctx, V2);                                       \
    Metadata *op2 = toMetadata(ctx, V3);                                       \
    Metadata *op3 = toMetadata(ctx, V4);                                       \
    Metadata *op4 = toMetadata(ctx, V5);                                       \
    return addAttr(m, ModuleAttrKind::NAME, {op0, op1, op2, op3, op4});        \
  }                                                                            \
  MODULE_GETTER(NAME, TY1, V1, 0)                                              \
  MODULE_GETTER(NAME, TY2, V2, 1)                                              \
  MODULE_GETTER(NAME, TY3, V3, 2)                                              \
  MODULE_GETTER(NAME, TY4, V4, 3)                                              \
  MODULE_GETTER(NAME, TY5, V5, 4)

#define MODULE_ATTR_6(NAME, IRNAME, TY1, V1, TY2, V2, TY3, V3, TY4, V4, TY5,   \
                      V5, TY6, V6)                                             \
  NamedMDNode &llvm::add##NAME##Attr(Module &m, TY1 V1, TY2 V2, TY3 V3,        \
                                     TY4 V4, TY5 V5, TY6 V6) {                 \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *op0 = toMetadata(ctx, V1);                                       \
    Metadata *op1 = toMetadata(ctx, V2);                                       \
    Metadata *op2 = toMetadata(ctx, V3);                                       \
    Metadata *op3 = toMetadata(ctx, V4);                                       \
    Metadata *op4 = toMetadata(ctx, V5);                                       \
    Metadata *op5 = toMetadata(ctx, V6);                                       \
    return addAttr(m, ModuleAttrKind::NAME, {op0, op1, op2, op3, op4, op5});   \
  }                                                                            \
  MODULE_GETTER(NAME, TY1, V1, 0)                                              \
  MODULE_GETTER(NAME, TY2, V2, 1)                                              \
  MODULE_GETTER(NAME, TY3, V3, 2)                                              \
  MODULE_GETTER(NAME, TY4, V4, 3)                                              \
  MODULE_GETTER(NAME, TY5, V5, 4)                                              \
  MODULE_GETTER(NAME, TY6, V6, 5)

#define MODULE_ATTR_7(NAME, IRNAME, TY1, V1, TY2, V2, TY3, V3, TY4, V4, TY5,   \
                      V5, TY6, V6, TY7, V7)                                    \
  NamedMDNode &llvm::add##NAME##Attr(Module &m, TY1 V1, TY2 V2, TY3 V3,        \
                                     TY4 V4, TY5 V5, TY6 V6, TY7 V7) {         \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *op0 = toMetadata(ctx, V1);                                       \
    Metadata *op1 = toMetadata(ctx, V2);                                       \
    Metadata *op2 = toMetadata(ctx, V3);                                       \
    Metadata *op3 = toMetadata(ctx, V4);                                       \
    Metadata *op4 = toMetadata(ctx, V5);                                       \
    Metadata *op5 = toMetadata(ctx, V6);                                       \
    Metadata *op6 = toMetadata(ctx, V7);                                       \
    return addAttr(m, ModuleAttrKind::NAME,                                    \
                   {op0, op1, op2, op3, op4, op5, op6});                       \
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
  NamedMDNode &llvm::add##NAME##Attr(Module &m, TY1 V1, TY2 V2, TY3 V3,        \
                                     TY4 V4, TY5 V5, TY6 V6, TY7 V7, TY8 V8) { \
    LLVMContext &ctx = m.getContext();                                         \
    Metadata *op0 = toMetadata(ctx, V1);                                       \
    Metadata *op1 = toMetadata(ctx, V2);                                       \
    Metadata *op2 = toMetadata(ctx, V3);                                       \
    Metadata *op3 = toMetadata(ctx, V4);                                       \
    Metadata *op4 = toMetadata(ctx, V5);                                       \
    Metadata *op5 = toMetadata(ctx, V6);                                       \
    Metadata *op6 = toMetadata(ctx, V7);                                       \
    Metadata *op7 = toMetadata(ctx, V8);                                       \
    return addAttr(m, ModuleAttrKind::NAME,                                    \
                   {op0, op1, op2, op3, op4, op5, op6, op7});                  \
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

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
#include "kitsune/Core/MetadataUtils.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/LoopInfo.h"

using namespace llvm;

template <typename T>
MDNode *llvm::getMetadataForAttr(LLVMContext &ctx, LoopAttrKind attr, T val) {
  StringRef name = getAttrName(attr);
  Metadata *mdVal = toMetadata(val, ctx);
  Metadata *mdTag = MDString::get(ctx, name);
  MDNode *md = MDNode::get(ctx, {mdTag, mdVal});

  return md;
}

MDNode *llvm::getMetadataForAttr(LLVMContext &ctx, LoopAttrKind attr) {
  return getMetadataForAttr(ctx, attr, 1U);
}

StringRef llvm::getAttrName(LoopAttrKind attr) {
  switch (attr) {
#define LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)                                  \
  case LoopAttrKind::NAME:                                                     \
    return IRNAME;

#define TAPIR_LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)                            \
  LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
  }
  llvm_unreachable("getAttrName: Attribute not handled");
}

std::optional<LoopAttrKind> llvm::getLoopAttrKind(StringRef name) {
  return StringSwitch<std::optional<LoopAttrKind>>(name)
#define LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE) .Case(IRNAME, LoopAttrKind::NAME)
#define TAPIR_LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)                            \
  LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
      .Default(std::nullopt);
}

bool llvm::isAttrTapirOnly(LoopAttrKind attr) {
  switch (attr) {
#define LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)                                  \
  case LoopAttrKind::NAME:                                                     \
    return false;
#define TAPIR_LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)                            \
  case LoopAttrKind::NAME:                                                     \
    return true;
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
  }
  llvm_unreachable("isLoopAttrTapirOnly: LoopAttrKind not handled");
}

bool llvm::hasAttr(const Loop &loop, LoopAttrKind attr) {
  return findOptionMDForLoop(&loop, getAttrName(attr));
}

template <typename T>
static std::optional<T> getAttr(const Loop &loop, StringRef name) {
  MDNode *md = findOptionMDForLoop(&loop, name);
  if (md && md->getNumOperands() == 2)
    return fromMetadata<T>(md->getOperand(1));
  return std::nullopt;
}

void llvm::removeAttr(Loop &loop, LoopAttrKind attr) {
  LLVMContext &ctx = loop.getHeader()->getContext();
  StringRef name = getAttrName(attr);
  MDNode *loopMD = loop.getLoopID();
  MDNode *newLoopMD = makePostTransformationMetadata(ctx, loopMD, {name}, {});

  loop.setLoopID(newLoopMD);
}

template <typename T>
static void addAttrAs(Loop &loop, LoopAttrKind attr, T val) {
  LLVMContext &ctx = loop.getHeader()->getContext();
  StringRef name = getAttrName(attr);
  MDNode *md = getMetadataForAttr(ctx, attr, val);
  MDNode *loopMD = loop.getLoopID();
  MDNode *newLoopMD = makePostTransformationMetadata(ctx, loopMD, {name}, {md});

  loop.setLoopID(newLoopMD);
}

static void addAttr(Loop &loop, LoopAttrKind attr) {
  LLVMContext &ctx = loop.getHeader()->getContext();
  StringRef name = getAttrName(attr);
  MDString *mdTag = MDString::get(ctx, name);
  MDNode *md = MDNode::get(ctx, mdTag);
  MDNode *loopMD = loop.getLoopID();
  MDNode *newLoopMD = makePostTransformationMetadata(ctx, loopMD, {name}, {md});

  loop.setLoopID(newLoopMD);
}

// Flag attributes (those that do not have a value) will have a different set of
// accessors. Mask them by defining LOOP_ATTRIBUTE_FLAG by defining it to an
// empty macro. These attributes may be applied to both tapir and regular loops.
#define LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)
#define LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)                                  \
  bool llvm::has##NAME##Attr(const Loop &loop) {                               \
    return hasAttr(loop, LoopAttrKind::NAME);                                  \
  }                                                                            \
                                                                               \
  std::optional<TYPE> llvm::get##NAME##Attr(const Loop &loop) {                \
    return getAttr<TYPE>(loop, IRNAME);                                        \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Loop &loop, TYPE val) {                           \
    addAttrAs(loop, LoopAttrKind::NAME, val);                                  \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(Loop &loop) {                                  \
    removeAttr(loop, LoopAttrKind::NAME);                                      \
  }
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

// Flag attributes (those that do not have a value) have a different set of
// accessors from non-flag attributes. These attributes may be applied to both
// tapir and regular loops.
#define LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                      \
  bool llvm::has##NAME##Attr(const Loop &loop) {                               \
    return hasAttr(loop, LoopAttrKind::NAME);                                  \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Loop &loop) {                                     \
    addAttrAs(loop, LoopAttrKind::NAME, 1U);                                   \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(Loop &loop) {                                  \
    removeAttr(loop, LoopAttrKind::NAME);                                      \
  }
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

// Flag attributes (those that do not have a value) will have a different set of
// accessors. Mask them by defining LOOP_ATTRIBUTE_FLAG by defining it to an
// empty macro. These attributes may be applied to tapir loops only.
#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)
#define TAPIR_LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)                            \
  bool llvm::has##NAME##Attr(const Loop &loop) {                               \
    return hasAttr(loop, LoopAttrKind::NAME);                                  \
  }                                                                            \
                                                                               \
  std::optional<TYPE> llvm::get##NAME##Attr(const Loop &loop) {                \
    return getAttr<TYPE>(loop, IRNAME);                                        \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Loop &loop, TYPE val) {                           \
    addAttrAs(loop, LoopAttrKind::NAME, val);                                  \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(Loop &loop) {                                  \
    removeAttr(loop, LoopAttrKind::NAME);                                      \
  }
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

// Flag attributes (those that do not have a value) have a different set of
// accessors from non-flag attributes. These attributes may be applied to tapir
// loops only.
#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                \
  bool llvm::has##NAME##Attr(const Loop &loop) {                               \
    return hasAttr(loop, LoopAttrKind::NAME);                                  \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(Loop &loop) {                                     \
    addAttr(loop, LoopAttrKind::NAME);                                         \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(Loop &loop) {                                  \
    removeAttr(loop, LoopAttrKind::NAME);                                      \
  }
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

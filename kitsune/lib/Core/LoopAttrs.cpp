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
MDNode *llvm::getMetadataForLoopAttr(LLVMContext &ctx, LoopAttrKind attr,
                                     T val) {
  StringRef name = getLoopAttrName(attr);
  Metadata *mdVal = toMetadata(ctx, val);
  Metadata *mdTag = MDString::get(ctx, name);
  MDNode *md = MDNode::get(ctx, {mdTag, mdVal});

  return md;
}

MDNode *llvm::getMetadataForLoopAttr(LLVMContext &ctx, LoopAttrKind attr) {
  return getMetadataForLoopAttr(ctx, attr, 1U);
}

StringRef llvm::getLoopAttrName(LoopAttrKind attr) {
  switch (attr) {
#define LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)                                  \
  case LoopAttrKind::NAME:                                                     \
    return IRNAME;

#define TAPIR_LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)                            \
  LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
  }
  llvm_unreachable("getLoopAttrName: Attribute not handled");
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

bool llvm::isLoopAttrTapirOnly(LoopAttrKind attr) {
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

bool llvm::hasLoopAttr(const Loop &loop, LoopAttrKind attr) {
  return findOptionMDForLoop(&loop, getLoopAttrName(attr));
}

template <typename T>
static std::optional<T> getLoopAttr(const Loop &loop, StringRef name) {
  MDNode *md = findOptionMDForLoop(&loop, name);
  if (md && md->getNumOperands() == 2)
    return fromMetadata<T>(md->getOperand(1));
  return std::nullopt;
}

void llvm::removeLoopAttr(Loop &loop, LoopAttrKind attr) {
  LLVMContext &ctx = loop.getHeader()->getContext();
  StringRef name = getLoopAttrName(attr);
  MDNode *loopMD = loop.getLoopID();
  MDNode *newLoopMD = makePostTransformationMetadata(ctx, loopMD, {name}, {});

  loop.setLoopID(newLoopMD);
}

template <typename T>
static void addLoopAttrAs(Loop &loop, LoopAttrKind attr, T val) {
  LLVMContext &ctx = loop.getHeader()->getContext();
  StringRef name = getLoopAttrName(attr);
  MDNode *md = getMetadataForLoopAttr(ctx, attr, val);
  MDNode *loopMD = loop.getLoopID();
  MDNode *newLoopMD = makePostTransformationMetadata(ctx, loopMD, {name}, {md});

  loop.setLoopID(newLoopMD);
}

// Flag attributes (those that do not have a value) will have a different set of
// accessors. Mask them by defining LOOP_ATTRIBUTE_FLAG by defining it to an
// empty macro. These attributes may be applied to both tapir and regular loops.
#define LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)
#define LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)                                  \
  bool llvm::hasLoop##NAME##Attr(const Loop &loop) {                           \
    return hasLoopAttr(loop, LoopAttrKind::NAME);                              \
  }                                                                            \
                                                                               \
  std::optional<TYPE> llvm::getLoop##NAME##Attr(const Loop &loop) {            \
    return getLoopAttr<TYPE>(loop, IRNAME);                                    \
  }                                                                            \
                                                                               \
  void llvm::addLoop##NAME##Attr(Loop &loop, TYPE val) {                       \
    addLoopAttrAs(loop, LoopAttrKind::NAME, val);                              \
  }                                                                            \
                                                                               \
  void llvm::removeLoop##NAME##Attr(Loop &loop) {                              \
    removeLoopAttr(loop, LoopAttrKind::NAME);                                  \
  }
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

// Flag attributes (those that do not have a value) have a different set of
// accessors from non-flag attributes. These attributes may be applied to both
// tapir and regular loops.
#define LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                      \
  bool llvm::hasLoop##NAME##Attr(const Loop &loop) {                           \
    return hasLoopAttr(loop, LoopAttrKind::NAME);                              \
  }                                                                            \
                                                                               \
  void llvm::addLoop##NAME##Attr(Loop &loop) {                                 \
    addLoopAttrAs(loop, LoopAttrKind::NAME, 1U);                               \
  }                                                                            \
                                                                               \
  void llvm::removeLoop##NAME##Attr(Loop &loop) {                              \
    removeLoopAttr(loop, LoopAttrKind::NAME);                                  \
  }
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

// Flag attributes (those that do not have a value) will have a different set of
// accessors. Mask them by defining LOOP_ATTRIBUTE_FLAG by defining it to an
// empty macro. These attributes may be applied to tapir loops only.
#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)
#define TAPIR_LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)                            \
  bool llvm::hasTapirLoop##NAME##Attr(const Loop &loop) {                      \
    return hasLoopAttr(loop, LoopAttrKind::NAME);                              \
  }                                                                            \
                                                                               \
  std::optional<TYPE> llvm::getTapirLoop##NAME##Attr(const Loop &loop) {       \
    return getLoopAttr<TYPE>(loop, IRNAME);                                    \
  }                                                                            \
                                                                               \
  void llvm::addTapirLoop##NAME##Attr(Loop &loop, TYPE val) {                  \
    addLoopAttrAs(loop, LoopAttrKind::NAME, val);                              \
  }                                                                            \
                                                                               \
  void llvm::removeTapirLoop##NAME##Attr(Loop &loop) {                         \
    removeLoopAttr(loop, LoopAttrKind::NAME);                                  \
  }
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

// Flag attributes (those that do not have a value) have a different set of
// accessors from non-flag attributes. These attributes may be applied to tapir
// loops only.
#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                \
  bool llvm::hasTapirLoop##NAME##Attr(const Loop &loop) {                      \
    return hasLoopAttr(loop, LoopAttrKind::NAME);                              \
  }                                                                            \
                                                                               \
  void llvm::addTapirLoop##NAME##Attr(Loop &loop) {                            \
    addLoopAttrAs(loop, LoopAttrKind::NAME, 1U);                               \
  }                                                                            \
                                                                               \
  void llvm::removeTapirLoop##NAME##Attr(Loop &loop) {                         \
    removeLoopAttr(loop, LoopAttrKind::NAME);                                  \
  }
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

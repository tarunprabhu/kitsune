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
#include "kitsune/Core/TypeUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"

using namespace llvm;

template <typename T>
static Constant *getConstantForValue(LLVMContext &ctx, T val) {
  if constexpr (std::is_enum_v<T>)
    return ConstantInt::get(getLLVMTypeFor<int32_t>(ctx), int32_t(val));
  else if constexpr (std::is_integral_v<T>)
    return ConstantInt::get(getLLVMTypeFor<T>(ctx), val);
  else if constexpr (std::is_same_v<StringRef, T>)
    return ConstantDataArray::getString(ctx, val, /*AddNull=*/false);
  else
    static_assert(0 && "Constant creation for type not implemented");
}

template <typename T>
MDNode *llvm::getMetadataForLoopAttr(LLVMContext &ctx, LoopAttrKind attr,
                                     T val) {
  StringRef name = getLoopAttrName(attr);
  Metadata *mdVal = ConstantAsMetadata::get(getConstantForValue(ctx, val));
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

template <typename Enum>
static std::optional<Enum> convertToEnum(const Constant &c) {
  if (const auto *cint = dyn_cast<ConstantInt>(&c)) {
    int v = cint->getLimitedValue();
    if constexpr (std::is_same_v<Enum, TTID>)
      return createTTIDFrom(v);
    else if constexpr (std::is_same_v<Enum, TapirSpawnStrategy>)
      return createTapirSpawnStrategyFrom(v);
    else
      static_assert(0 && "Enum value not handled");
  }
  return std::nullopt;
}

template <typename T>
static std::optional<T> convertToIntegral(const Constant &c) {
  if (const auto *cint = dyn_cast<ConstantInt>(&c))
    return cint->getLimitedValue();
  return std::nullopt;
}

[[maybe_unused]]
static std::optional<StringRef> convertToStringRef(const Constant &c) {
  if (const auto *cda = dyn_cast<ConstantDataArray>(&c)) {
    if (cda->isString())
      return cda->getAsString();
    else if (cda->isCString())
      return cda->getAsCString();
  }
  return std::nullopt;
}

template <typename T> std::optional<T> convertTo(const Constant &c) {
  if constexpr (std::is_enum_v<T>)
    return convertToEnum<T>(c);
  else if constexpr (std::is_integral_v<T>)
    return convertToIntegral<T>(c);
  else if constexpr (std::is_same_v<T, StringRef>)
    return convertToStringRef(c);
}

template <typename T>
static std::optional<T> getLoopAttr(const Loop &loop, StringRef name) {
  MDNode *md = findOptionMDForLoop(&loop, name);
  if (md && md->getNumOperands() == 2)
    if (auto *cmd = dyn_cast<ConstantAsMetadata>(md->getOperand(1)))
      if (auto *c = dyn_cast<Constant>(cmd->getValue()))
        return convertTo<T>(*c);
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

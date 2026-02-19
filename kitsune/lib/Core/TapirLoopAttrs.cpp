//===- TapirLoopAttrs.cpp - Tapir loop attributes and utilities -----------===//
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

#include "kitsune/Core/TapirLoopAttrs.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"

using namespace llvm;

template <typename T>
static Constant *getConstantForValue(LLVMContext &ctx, T val) {
  if constexpr (std::is_enum_v<T>)
    return ConstantInt::get(Type::getInt32Ty(ctx), unsigned(val));
  else if constexpr (std::is_integral_v<T>)
    return ConstantInt::get(IntegerType::get(ctx, sizeof(T) * 8), val);
  else if constexpr (std::is_same_v<StringRef, T>)
    return ConstantDataArray::getString(ctx, val, /*AddNull=*/false);
  else
    static_assert(0 && "Constant creation for type not implemented");
}

template <typename T>
static MDNode *getMDNodeForAttr(LLVMContext &ctx, StringRef name, T val) {
  Metadata *mdVal = ConstantAsMetadata::get(getConstantForValue(ctx, val));
  Metadata *mdTag = MDString::get(ctx, name);
  MDNode *md = MDNode::get(ctx, {mdTag, mdVal});

  return md;
}

MDNode *llvm::getMetadataForTapirLoopAttr(LLVMContext &ctx,
                                          TapirLoopAttrKind attr) {
  return getMDNodeForAttr(ctx, getTapirLoopAttrName(attr), 1U);
}

template <typename T>
MDNode *llvm::getMetadataForTapirLoopAttr(LLVMContext &ctx,
                                          TapirLoopAttrKind attr, T val) {
  return getMDNodeForAttr(ctx, getTapirLoopAttrName(attr), val);
}

// Explicit instantiations for attributes.
template MDNode *llvm::getMetadataForTapirLoopAttr(LLVMContext &ctx,
                                                   TapirLoopAttrKind attr,
                                                   StringRef val);
template MDNode *llvm::getMetadataForTapirLoopAttr(LLVMContext &ctx,
                                                   TapirLoopAttrKind attr,
                                                   unsigned val);
template MDNode *llvm::getMetadataForTapirLoopAttr(LLVMContext &ctx,
                                                   TapirLoopAttrKind attr,
                                                   unsigned long val);
#define TAPIR_LOOP_ATTRIBUTE_ENUM(NAME, IRNAME, TYPE)                          \
  template MDNode *llvm::getMetadataForTapirLoopAttr(                          \
      LLVMContext &ctx, TapirLoopAttrKind attr, TYPE val);
#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_STR(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_UINT(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_ULONG(NAME, IRNAME)

#include "kitsune/Core/TapirLoopAttrs.inc"

#undef TAPIR_LOOP_ATTRIBUTE_ULONG
#undef TAPIR_LOOP_ATTRIBUTE_UINT
#undef TAPIR_LOOP_ATTRIBUTE_STR
#undef TAPIR_LOOP_ATTRIBUTE_FLAG
#undef TAPIR_LOOP_ATTRIBUTE_ENUM

StringRef llvm::getTapirLoopAttrName(TapirLoopAttrKind attr) {
#define TAPIR_LOOP_ATTRIBUTE_ENUM(NAME, IRNAME, TYPE)                          \
  case TapirLoopAttrKind::NAME:                                                \
    return IRNAME;

#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                \
  case TapirLoopAttrKind::NAME:                                                \
    return IRNAME;

#define TAPIR_LOOP_ATTRIBUTE_STR(NAME, IRNAME)                                 \
  case TapirLoopAttrKind::NAME:                                                \
    return IRNAME;

#define TAPIR_LOOP_ATTRIBUTE_UINT(NAME, IRNAME)                                \
  case TapirLoopAttrKind::NAME:                                                \
    return IRNAME;

#define TAPIR_LOOP_ATTRIBUTE_ULONG(NAME, IRNAME)                               \
  case TapirLoopAttrKind::NAME:                                                \
    return IRNAME;

  switch (attr) {
#include "kitsune/Core/TapirLoopAttrs.inc"
  }
  llvm_unreachable("getTapirLoopAttrName: Attribute not handled");

#undef TAPIR_LOOP_ATTRIBUTE_ULONG
#undef TAPIR_LOOP_ATTRIBUTE_UINT
#undef TAPIR_LOOP_ATTRIBUTE_STR
#undef TAPIR_LOOP_ATTRIBUTE_FLAG
#undef TAPIR_LOOP_ATTRIBUTE_ENUM
#undef TAPIR_LOOP_ATTR
}

bool llvm::hasTapirLoopAttr(const Loop &loop, TapirLoopAttrKind attr) {
#define TAPIR_LOOP_ATTR(NAME, IRNAME)                                          \
  case TapirLoopAttrKind::NAME:                                                \
    return findOptionMDForLoop(&loop, IRNAME);

#define TAPIR_LOOP_ATTRIBUTE_ENUM(NAME, IRNAME, TYPE)                          \
  TAPIR_LOOP_ATTR(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME) TAPIR_LOOP_ATTR(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_STR(NAME, IRNAME) TAPIR_LOOP_ATTR(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_UINT(NAME, IRNAME) TAPIR_LOOP_ATTR(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_ULONG(NAME, IRNAME) TAPIR_LOOP_ATTR(NAME, IRNAME)

  switch (attr) {
#include "kitsune/Core/TapirLoopAttrs.inc"
  }
  llvm_unreachable("hasTapirLoopAttr: Attribute not handled");

#undef TAPIR_LOOP_ATTRIBUTE_ULONG
#undef TAPIR_LOOP_ATTRIBUTE_UINT
#undef TAPIR_LOOP_ATTRIBUTE_STR
#undef TAPIR_LOOP_ATTRIBUTE_FLAG
#undef TAPIR_LOOP_ATTRIBUTE_ENUM
#undef TAPIR_LOOP_ATTR
}

template <typename Enum>
static std::optional<Enum> convertToEnum(const Constant &c) {
  if (const auto *cint = dyn_cast<ConstantInt>(&c)) {
    unsigned v = cint->getLimitedValue();
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
static std::optional<T> getTapirLoopAttr(const Loop &loop, StringRef name) {
  MDNode *md = findOptionMDForLoop(&loop, name);
  if (md && md->getNumOperands() == 2)
    if (auto *cmd = dyn_cast<ConstantAsMetadata>(md->getOperand(1)))
      if (auto *c = dyn_cast<Constant>(cmd->getValue()))
        return convertTo<T>(*c);
  return std::nullopt;
}

static void removeTapirLoopAttrWithName(Loop &loop, StringRef name) {
  LLVMContext &ctx = loop.getHeader()->getContext();
  MDNode *loopMD = loop.getLoopID();
  MDNode *newLoopMD = makePostTransformationMetadata(ctx, loopMD, {name}, {});

  loop.setLoopID(newLoopMD);
}

void llvm::removeTapirLoopAttr(Loop &loop, TapirLoopAttrKind attr) {
#define TAPIR_LOOP_ATTR(NAME, IRNAME)                                          \
  case TapirLoopAttrKind::NAME:                                                \
    return removeTapirLoopAttrWithName(loop, IRNAME);

#define TAPIR_LOOP_ATTRIBUTE_ENUM(NAME, IRNAME, TYPE)                          \
  TAPIR_LOOP_ATTR(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME) TAPIR_LOOP_ATTR(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_STR(NAME, IRNAME) TAPIR_LOOP_ATTR(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_UINT(NAME, IRNAME) TAPIR_LOOP_ATTR(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_ULONG(NAME, IRNAME) TAPIR_LOOP_ATTR(NAME, IRNAME)

  switch (attr) {
#include "kitsune/Core/TapirLoopAttrs.inc"
  }
  llvm_unreachable("removeTapirLoopAttr: Attribute not handled");

#undef TAPIR_LOOP_ATTRIBUTE_ULONG
#undef TAPIR_LOOP_ATTRIBUTE_UINT
#undef TAPIR_LOOP_ATTRIBUTE_STR
#undef TAPIR_LOOP_ATTRIBUTE_FLAG
#undef TAPIR_LOOP_ATTRIBUTE_ENUM
#undef TAPIR_LOOP_ATTR
}

template <typename T>
static void addTapirLoopAttrAs(Loop &loop, StringRef name, T val) {
  LLVMContext &ctx = loop.getHeader()->getContext();
  MDNode *md = getMDNodeForAttr(ctx, name, val);
  MDNode *loopMD = loop.getLoopID();
  MDNode *newLoopMD = makePostTransformationMetadata(ctx, loopMD, {name}, {md});

  loop.setLoopID(newLoopMD);
}

void llvm::clearTapirLoopAttrs(Loop &loop) {
  LLVMContext &ctx = loop.getHeader()->getContext();
  MDNode *loopMD = loop.getLoopID();
  MDNode *newLoopMD = makePostTransformationMetadata(
      ctx, loopMD, {tapirLoopAttrNamePrefix}, {});

  loop.setLoopID(newLoopMD);
}

#define TAPIR_LOOP_ATTRIBUTE(NAME, IRNAME, TYPE, IRTYPE)                       \
  bool llvm::hasTapirLoop##NAME##Attr(const Loop &loop) {                      \
    return hasTapirLoopAttr(loop, TapirLoopAttrKind::NAME);                    \
  }                                                                            \
                                                                               \
  std::optional<TYPE> llvm::getTapirLoop##NAME##Attr(const Loop &loop) {       \
    return getTapirLoopAttr<TYPE>(loop, IRNAME);                               \
  }                                                                            \
                                                                               \
  void llvm::addTapirLoop##NAME##Attr(Loop &loop, TYPE val) {                  \
    addTapirLoopAttrAs(loop, IRNAME, IRTYPE(val));                             \
  }                                                                            \
                                                                               \
  void llvm::removeTapirLoop##NAME##Attr(Loop &loop) {                         \
    removeTapirLoopAttr(loop, TapirLoopAttrKind::NAME);                        \
  }

#define TAPIR_LOOP_ATTRIBUTE_ENUM(NAME, IRNAME, TYPE)                          \
  TAPIR_LOOP_ATTRIBUTE(NAME, IRNAME, TYPE, unsigned)

#define TAPIR_LOOP_ATTRIBUTE_STR(NAME, IRNAME)                                 \
  TAPIR_LOOP_ATTRIBUTE(NAME, IRNAME, StringRef, StringRef)

#define TAPIR_LOOP_ATTRIBUTE_UINT(NAME, IRNAME)                                \
  TAPIR_LOOP_ATTRIBUTE(NAME, IRNAME, unsigned, unsigned)

#define TAPIR_LOOP_ATTRIBUTE_ULONG(NAME, IRNAME)                               \
  TAPIR_LOOP_ATTRIBUTE(NAME, IRNAME, unsigned long, unsigned long)

#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                \
  bool llvm::hasTapirLoop##NAME##Attr(const Loop &loop) {                      \
    return hasTapirLoopAttr(loop, TapirLoopAttrKind::NAME);                    \
  }                                                                            \
                                                                               \
  void llvm::addTapirLoop##NAME##Attr(Loop &loop) {                            \
    addTapirLoopAttrAs(loop, IRNAME, 1U);                                      \
  }                                                                            \
                                                                               \
  void llvm::removeTapirLoop##NAME##Attr(Loop &loop) {                         \
    removeTapirLoopAttr(loop, TapirLoopAttrKind::NAME);                        \
  }

#include "kitsune/Core/TapirLoopAttrs.inc"

#undef TAPIR_LOOP_ATTRIBUTE_ULONG
#undef TAPIR_LOOP_ATTRIBUTE_UINT
#undef TAPIR_LOOP_ATTRIBUTE_STR
#undef TAPIR_LOOP_ATTRIBUTE_FLAG
#undef TAPIR_LOOP_ATTRIBUTE_ENUM
#undef TAPIR_LOOP_ATTRIBUTE

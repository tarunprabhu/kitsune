//=- AttrsImpl.h - Common definitions Kitsune-specific attributes -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common definitions for Kitsune-specific attributes.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_CORE_ATTRS_IMPL_H
#define KITSUNE_LIB_CORE_ATTRS_IMPL_H

#include "kitsune/Core/AttrsCommon.h"
#include "kitsune/Core/VerifierInternal.h"
#include "kitsune/Support/ToString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LoopInfo.h"

#define DEFN_ATTR_GENERIC(IRELEM, ENUMKIND)                                    \
  static void setAttrList(IRELEM &, MDNode *attrList);                         \
  static void addAttr(IRELEM &, StringRef name, ArrayRef<Metadata *> vals);    \
  static void removeAttr(IRELEM &, StringRef attrName);                        \
                                                                               \
  bool llvm::hasAttr(const IRELEM &ir, ENUMKIND attr) {                        \
    return getRawAttr(getAttrName(attr), getRawAttrList(ir));                  \
  }                                                                            \
                                                                               \
  void llvm::removeAttr(IRELEM &ir, ENUMKIND attr) {                           \
    ::removeAttr(ir, getAttrName(attr));                                       \
  }                                                                            \
                                                                               \
  iterator_range<AttrIterator> llvm::attrs(const IRELEM &ir) {                 \
    if (const MDNode *attrList = getRawAttrList(ir)) {                         \
      AttrIterator beg(attrList);                                              \
      AttrIterator end(attrList, attrList->getNumOperands());                  \
                                                                               \
      return iterator_range(beg, end);                                         \
    }                                                                          \
    return iterator_range(AttrIterator(), AttrIterator());                     \
  }

#define DEFN_ATTR_COMMON(IRELEM, ENUMKIND, NAME, IRNAME, CUSTOMVERIFY, TYPE)   \
  bool llvm::has##NAME##Attr(const IRELEM &ir) {                               \
    return getRawAttr(IRNAME, getRawAttrList(ir));                             \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(IRELEM &ir) { ::removeAttr(ir, IRNAME); }

#define DEFN_ATTR_LOOP(IRELEM, NAME, IRNAME, CUSTOMVERIFY)                     \
  std::optional<Loop *> llvm::get##NAME##Attr(                                 \
      const IRELEM &ir, const SmallVectorImpl<const LoopInfo *> &lis) {        \
    return getAttrValue(IRNAME, getRawAttrList(ir), lis);                      \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(IRELEM &ir, const Loop &loop) {                   \
    ::addAttr(ir, IRNAME, loop.getLoopID());                                   \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(KitVerifier &v, const IRELEM &ir) {            \
    v.push();                                                                  \
    if (const MDNode *attr = getRawAttr(IRNAME, getRawAttrList(ir))) {         \
      if (!detail::verifyRawAttrValueCount(v, *attr, 1))                       \
        return v.pop();                                                        \
                                                                               \
      if (!detail::verifyRawAttrValueLoop(v, *attr))                           \
        return v.pop();                                                        \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        verify##NAME##Attr(v, ir, *cast<MDNode>(attr->getOperand(1)));         \
    }                                                                          \
    return v.pop();                                                            \
  }

#define DEFN_ATTR_0(IRELEM, NAME, IRNAME, CUSTOMVERIFY)                        \
  void llvm::add##NAME##Attr(IRELEM &ir) { ::addAttr(ir, IRNAME, {}); }        \
                                                                               \
  bool llvm::verify##NAME##Attr(KitVerifier &v, const IRELEM &ir) {            \
    v.push();                                                                  \
    if (const MDNode *attr = getRawAttr(IRNAME, getRawAttrList(ir))) {         \
      if (!detail::verifyRawAttrValueCount(v, *attr, 0))                       \
        return v.pop();                                                        \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        verify##NAME##Attr(v, ir, true);                                       \
    }                                                                          \
    return v.pop();                                                            \
  }

#define DEFN_ATTR_1(IRELEM, NAME, IRNAME, CUSTOMVERIFY, TYPE)                  \
  std::optional<TYPE> llvm::get##NAME##Attr(const IRELEM &ir) {                \
    return getAttrValue<TYPE>(IRNAME, getRawAttrList(ir), 0, 1);               \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(IRELEM &ir, const TYPE &val) {                    \
    LLVMContext &ctx = getContext(ir);                                         \
    Metadata *attrVals[] = {toMetadata(val, ctx)};                             \
    ::addAttr(ir, IRNAME, attrVals);                                           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(KitVerifier &v, const IRELEM &ir) {            \
    v.push();                                                                  \
    if (const MDNode *attr = getRawAttr(IRNAME, getRawAttrList(ir))) {         \
      if (!detail::verifyRawAttrValueCount(v, *attr, 1))                       \
        return v.pop();                                                        \
                                                                               \
      std::optional<TYPE> val = getRawAttrValue<TYPE>(*attr, 0);               \
      if (!detail::verifyRawAttrValues(v, *attr, val))                         \
        return v.pop();                                                        \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        verify##NAME##Attr(v, ir, *val);                                       \
    }                                                                          \
    return v.pop();                                                            \
  }

#define DEFN_ATTR_2(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1)                                         \
  void llvm::add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1) {     \
    LLVMContext &ctx = getContext(ir);                                         \
    Metadata *attrVals[] = {toMetadata(e0, ctx), toMetadata(e1, ctx)};         \
    ::addAttr(ir, IRNAME, attrVals);                                           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(KitVerifier &v, const IRELEM &ir) {            \
    v.push();                                                                  \
    if (const MDNode *attr = getRawAttr(IRNAME, getRawAttrList(ir))) {         \
      if (!detail::verifyRawAttrValueCount(v, *attr, 2))                       \
        return v.pop();                                                        \
                                                                               \
      std::optional<ETY0> v0 = getRawAttrValue<ETY0>(*attr, 0);                \
      std::optional<ETY1> v1 = getRawAttrValue<ETY1>(*attr, 1);                \
      if (!detail::verifyRawAttrValues(v, *attr, v0, v1))                      \
        return v.pop();                                                        \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        verify##NAME##Attr(v, ir, *v0, *v1);                                   \
    }                                                                          \
    return v.pop();                                                            \
  }

#define DEFN_ATTR_3(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2)                      \
  void llvm::add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,       \
                             const ETY2 &e2) {                                 \
    LLVMContext &ctx = getContext(ir);                                         \
    Metadata *attrVals[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),          \
                            toMetadata(e2, ctx)};                              \
    ::addAttr(ir, IRNAME, attrVals);                                           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(KitVerifier &v, const IRELEM &ir) {            \
    v.push();                                                                  \
    if (const MDNode *attr = getRawAttr(IRNAME, getRawAttrList(ir))) {         \
      if (!detail::verifyRawAttrValueCount(v, *attr, 3))                       \
        return v.pop();                                                        \
                                                                               \
      std::optional<ETY0> v0 = getRawAttrValue<ETY0>(*attr, 0);                \
      std::optional<ETY1> v1 = getRawAttrValue<ETY1>(*attr, 1);                \
      std::optional<ETY2> v2 = getRawAttrValue<ETY2>(*attr, 2);                \
      if (!detail::verifyRawAttrValues(v, *attr, v0, v1, v2))                  \
        return v.pop();                                                        \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        verify##NAME##Attr(v, ir, *v0, *v1, *v2);                              \
    }                                                                          \
    return v.pop();                                                            \
  }

#define DEFN_ATTR_4(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3)   \
  void llvm::add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,       \
                             const ETY2 &e2, const ETY3 &e3) {                 \
    LLVMContext &ctx = getContext(ir);                                         \
    Metadata *attrVals[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),          \
                            toMetadata(e2, ctx), toMetadata(e3, ctx)};         \
    ::addAttr(ir, IRNAME, attrVals);                                           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(KitVerifier &v, const IRELEM &ir) {            \
    v.push();                                                                  \
    if (const MDNode *attr = getRawAttr(IRNAME, getRawAttrList(ir))) {         \
      if (!detail::verifyRawAttrValueCount(v, *attr, 4))                       \
        return v.pop();                                                        \
                                                                               \
      std::optional<ETY0> v0 = getRawAttrValue<ETY0>(*attr, 0);                \
      std::optional<ETY1> v1 = getRawAttrValue<ETY1>(*attr, 1);                \
      std::optional<ETY2> v2 = getRawAttrValue<ETY2>(*attr, 2);                \
      std::optional<ETY3> v3 = getRawAttrValue<ETY3>(*attr, 3);                \
      if (!detail::verifyRawAttrValues(v, *attr, v0, v1, v2, v3))              \
        return v.pop();                                                        \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        verify##NAME##Attr(v, ir, *v0, *v1, *v2, *v3);                         \
    }                                                                          \
    return v.pop();                                                            \
  }

#define DEFN_ATTR_5(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4)                                         \
  void llvm::add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,       \
                             const ETY2 &e2, const ETY3 &e3, const ETY4 &e4) { \
    LLVMContext &ctx = getContext(ir);                                         \
    Metadata *attrVals[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),          \
                            toMetadata(e2, ctx), toMetadata(e3, ctx),          \
                            toMetadata(e4, ctx)};                              \
    ::addAttr(ir, IRNAME, attrVals);                                           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(KitVerifier &v, const IRELEM &ir) {            \
    v.push();                                                                  \
    if (const MDNode *attr = getRawAttr(IRNAME, getRawAttrList(ir))) {         \
      if (!detail::verifyRawAttrValueCount(v, *attr, 5))                       \
        return v.pop();                                                        \
                                                                               \
      std::optional<ETY0> v0 = getRawAttrValue<ETY0>(*attr, 0);                \
      std::optional<ETY1> v1 = getRawAttrValue<ETY1>(*attr, 1);                \
      std::optional<ETY2> v2 = getRawAttrValue<ETY2>(*attr, 2);                \
      std::optional<ETY3> v3 = getRawAttrValue<ETY3>(*attr, 3);                \
      std::optional<ETY4> v4 = getRawAttrValue<ETY4>(*attr, 4);                \
      if (!detail::verifyRawAttrValues(v, *attr, v0, v1, v2, v3, v4))          \
        return v.pop();                                                        \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        verify##NAME##Attr(v, ir, *v0, *v1, *v2, *v3, *v4);                    \
    }                                                                          \
    return v.pop();                                                            \
  }

#define DEFN_ATTR_6(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4, ETY5, ENAME5, EN5)                      \
  void llvm::add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,       \
                             const ETY2 &e2, const ETY3 &e3, const ETY4 &e4,   \
                             const ETY5 &e5) {                                 \
    LLVMContext &ctx = getContext(ir);                                         \
    Metadata *attrVals[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),          \
                            toMetadata(e2, ctx), toMetadata(e3, ctx),          \
                            toMetadata(e4, ctx), toMetadata(e5, ctx)};         \
    ::addAttr(ir, IRNAME, attrVals);                                           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(KitVerifier &v, const IRELEM &ir) {            \
    v.push();                                                                  \
    if (const MDNode *attr = getRawAttr(IRNAME, getRawAttrList(ir))) {         \
      if (!detail::verifyRawAttrValueCount(v, *attr, 6))                       \
        return v.pop();                                                        \
                                                                               \
      std::optional<ETY0> v0 = getRawAttrValue<ETY0>(*attr, 0);                \
      std::optional<ETY1> v1 = getRawAttrValue<ETY1>(*attr, 1);                \
      std::optional<ETY2> v2 = getRawAttrValue<ETY2>(*attr, 2);                \
      std::optional<ETY3> v3 = getRawAttrValue<ETY3>(*attr, 3);                \
      std::optional<ETY4> v4 = getRawAttrValue<ETY4>(*attr, 4);                \
      std::optional<ETY5> v5 = getRawAttrValue<ETY5>(*attr, 5);                \
      if (!detail::verifyRawAttrValues(v, *attr, v0, v1, v2, v3, v4, v5))      \
        return v.pop();                                                        \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        verify##NAME##Attr(v, ir, *v0, *v1, *v2, *v3, *v4, *v5);               \
    }                                                                          \
    return v.pop();                                                            \
  }

#define DEFN_ATTR_7(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4, ETY5, ENAME5, EN5, ETY6, ENAME6, EN6)   \
  void llvm::add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,       \
                             const ETY2 &e2, const ETY3 &e3, const ETY4 &e4,   \
                             const ETY5 &e5, const ETY6 &e6) {                 \
    LLVMContext &ctx = getContext(ir);                                         \
    Metadata *attrVals[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),          \
                            toMetadata(e2, ctx), toMetadata(e3, ctx),          \
                            toMetadata(e4, ctx), toMetadata(e5, ctx),          \
                            toMetadata(e6, ctx)};                              \
    ::addAttr(ir, IRNAME, attrVals);                                           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(KitVerifier &v, const IRELEM &ir) {            \
    v.push();                                                                  \
    if (const MDNode *attr = getRawAttr(IRNAME, getRawAttrList(ir))) {         \
      if (!detail::verifyRawAttrValueCount(v, *attr, 7))                       \
        return v.pop();                                                        \
                                                                               \
      std::optional<ETY0> v0 = getRawAttrValue<ETY0>(*attr, 0);                \
      std::optional<ETY1> v1 = getRawAttrValue<ETY1>(*attr, 1);                \
      std::optional<ETY2> v2 = getRawAttrValue<ETY2>(*attr, 2);                \
      std::optional<ETY3> v3 = getRawAttrValue<ETY3>(*attr, 3);                \
      std::optional<ETY4> v4 = getRawAttrValue<ETY4>(*attr, 4);                \
      std::optional<ETY5> v5 = getRawAttrValue<ETY5>(*attr, 5);                \
      std::optional<ETY6> v6 = getRawAttrValue<ETY6>(*attr, 6);                \
      if (!detail::verifyRawAttrValues(v, *attr, v0, v1, v2, v3, v4, v5, v6))  \
        return v.pop();                                                        \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        verify##NAME##Attr(v, ir, *v0, *v1, *v2, *v3, *v4, *v5, *v6);          \
    }                                                                          \
    return v.pop();                                                            \
  }

#define DEFN_ATTR_8(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4, ETY5, ENAME5, EN5, ETY6, ENAME6, EN6,   \
                    ETY7, ENAME7, EN7)                                         \
  void llvm::add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,       \
                             const ETY2 &e2, const ETY3 &e3, const ETY4 &e4,   \
                             const ETY5 &e5, const ETY6 &e6, const ETY7 &e7) { \
    LLVMContext &ctx = getContext(ir);                                         \
    Metadata *attrVals[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),          \
                            toMetadata(e2, ctx), toMetadata(e3, ctx),          \
                            toMetadata(e4, ctx), toMetadata(e5, ctx),          \
                            toMetadata(e6, ctx), toMetadata(e7, ctx)};         \
    ::addAttr(ir, IRNAME, attrVals);                                           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(KitVerifier &v, const IRELEM &ir) {            \
    v.push();                                                                  \
    if (const MDNode *attr = getRawAttr(IRNAME, getRawAttrList(ir))) {         \
      if (!detail::verifyRawAttrValueCount(v, *attr, 8))                       \
        return v.pop();                                                        \
                                                                               \
      std::optional<ETY0> v0 = getRawAttrValue<ETY0>(*attr, 0);                \
      std::optional<ETY1> v1 = getRawAttrValue<ETY1>(*attr, 1);                \
      std::optional<ETY2> v2 = getRawAttrValue<ETY2>(*attr, 2);                \
      std::optional<ETY3> v3 = getRawAttrValue<ETY3>(*attr, 3);                \
      std::optional<ETY4> v4 = getRawAttrValue<ETY4>(*attr, 4);                \
      std::optional<ETY5> v5 = getRawAttrValue<ETY5>(*attr, 5);                \
      std::optional<ETY6> v6 = getRawAttrValue<ETY6>(*attr, 6);                \
      std::optional<ETY7> v7 = getRawAttrValue<ETY7>(*attr, 7);                \
      if (!detail::verifyRawAttrValues(v, *attr, v0, v1, v2, v3, v4, v5, v6,   \
                                       v7))                                    \
        return v.pop();                                                        \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        verify##NAME##Attr(v, ir, *v0, *v1, *v2, *v3, *v4, *v5, *v6, *v7);     \
    }                                                                          \
    return v.pop();                                                            \
  }

#define DEFN_ATTR_N(IRELEM, NAME, IRNAME, ETY, ENAME, EN, NELEMS)              \
  std::optional<ETY> llvm::get##ENAME##From##NAME##Attr(const IRELEM &ir) {    \
    return getAttrValue<ETY>(IRNAME, getRawAttrList(ir), EN, NELEMS);          \
  }

#endif // KITSUNE_LIB_CORE_ATTRS_IMPL_H

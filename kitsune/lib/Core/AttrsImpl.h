//===- AttrsImpl.h - Common definitions for Kitsune-specific attributes ---===//
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

#include "VerifyImpl.h"
#include "kitsune/Core/AttrsCommon.h"
#include "kitsune/Support/ToString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LoopInfo.h"

#define VERIFY_IMPL(V, OS, ETYPE, ENAME, EN, IRNAME)                           \
  detail::check(V.has_value(), OS, detail::errMsgNoValueAt, toString<ETYPE>(), \
                #ENAME, EN, IRNAME)

#define DEFN_ATTR_GENERIC(IRELEM, ENUMKIND)                                    \
  static void setAttrList(IRELEM &, MDNode *attrList);                         \
  static void addAttr(IRELEM &, StringRef name, ArrayRef<Metadata *> vals);    \
  static void removeAttr(IRELEM &, StringRef attrName);                        \
                                                                               \
  bool llvm::hasAttr(const IRELEM &ir, ENUMKIND attr) {                        \
    return getRawAttr(getAttrName(attr), getAttrList(ir));                     \
  }                                                                            \
                                                                               \
  void llvm::removeAttr(IRELEM &ir, ENUMKIND attr) {                           \
    ::removeAttr(ir, getAttrName(attr));                                       \
  }

#define DEFN_ATTR_COMMON(IRELEM, ENUMKIND, NAME, IRNAME, CUSTOMVERIFY, TYPE)   \
  bool llvm::has##NAME##Attr(const IRELEM &ir) {                               \
    return getRawAttr(IRNAME, getAttrList(ir));                                \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(IRELEM &ir) { ::removeAttr(ir, IRNAME); }

#define DEFN_ATTR_LOOP(IRELEM, NAME, IRNAME, CUSTOMVERIFY)                     \
  std::optional<Loop *> llvm::get##NAME##Attr(                                 \
      const IRELEM &ir, const SmallVectorImpl<const LoopInfo *> &lis) {        \
    return getAttrValue(IRNAME, getAttrList(ir), lis);                         \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(IRELEM &ir, const Loop &loop) {                   \
    ::addAttr(ir, IRNAME, loop.getLoopID());                                   \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const IRELEM &ir, raw_ostream *os) {           \
    bool isValid = true;                                                       \
    if (has##NAME##Attr(ir)) {                                                 \
      isValid &= verifyAttrLoop(IRNAME, getAttrList(ir), os);                  \
                                                                               \
      if constexpr (CUSTOMVERIFY) {                                            \
        MDNode *raw = getRawAttr(IRNAME, getAttrList(ir));                     \
        MDNode *loopID = cast<MDNode>(raw->getOperand(1));                     \
        isValid &= verify##NAME##Attr(ir, *loopID, os);                        \
      }                                                                        \
    }                                                                          \
    return isValid;                                                            \
  }

#define DEFN_ATTR_0(IRELEM, NAME, IRNAME, CUSTOMVERIFY)                        \
  void llvm::add##NAME##Attr(IRELEM &ir) { ::addAttr(ir, IRNAME, {}); }        \
                                                                               \
  bool llvm::verify##NAME##Attr(const IRELEM &ir, raw_ostream *os) {           \
    bool isValid = true;                                                       \
    if (has##NAME##Attr(ir)) {                                                 \
      isValid &= verifyAttr0(IRNAME, getAttrList(ir), os);                     \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        isValid &= verify##NAME##Attr(ir, true, os);                           \
    }                                                                          \
    return isValid;                                                            \
  }

#define DEFN_ATTR_1(IRELEM, NAME, IRNAME, CUSTOMVERIFY, TYPE)                  \
  std::optional<TYPE> llvm::get##NAME##Attr(const IRELEM &ir) {                \
    return getAttrValue<TYPE>(IRNAME, getAttrList(ir), 0, 1);                  \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(IRELEM &ir, const TYPE &val) {                    \
    LLVMContext &ctx = getContext(ir);                                         \
    Metadata *attrVals[] = {toMetadata(val, ctx)};                             \
    ::addAttr(ir, IRNAME, attrVals);                                           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const IRELEM &ir, raw_ostream *os) {           \
    bool isValid = true;                                                       \
    if (has##NAME##Attr(ir)) {                                                 \
      std::optional<TYPE> v = get##NAME##Attr(ir);                             \
      isValid &=                                                               \
          detail::check(v.has_value(), os,                                     \
                        "Could not get value of type '{}' in attribute '{}'",  \
                        toString<TYPE>(), IRNAME);                             \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        isValid &= verify##NAME##Attr(ir, *v, os);                             \
    }                                                                          \
    return isValid;                                                            \
  }

#define DEFN_ATTR_2(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1)                                         \
  void llvm::add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1) {     \
    LLVMContext &ctx = getContext(ir);                                         \
    Metadata *attrVals[] = {toMetadata(e0, ctx), toMetadata(e1, ctx)};         \
    ::addAttr(ir, IRNAME, attrVals);                                           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const IRELEM &ir, raw_ostream *os) {           \
    bool isValid = true;                                                       \
    if (has##NAME##Attr(ir)) {                                                 \
      std::optional<ETY0> v0 = get##ENAME0##From##NAME##Attr(ir);              \
      std::optional<ETY1> v1 = get##ENAME1##From##NAME##Attr(ir);              \
                                                                               \
      isValid &= VERIFY_IMPL(v0, os, ETY0, ENAME0, EN0, IRNAME);               \
      isValid &= VERIFY_IMPL(v1, os, ETY1, ENAME1, EN1, IRNAME);               \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        isValid &= verify##NAME##Attr(ir, *v0, *v1, os);                       \
    }                                                                          \
    return isValid;                                                            \
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
  bool llvm::verify##NAME##Attr(const IRELEM &ir, raw_ostream *os) {           \
    bool isValid = true;                                                       \
    if (has##NAME##Attr(ir)) {                                                 \
      std::optional<ETY0> v0 = get##ENAME0##From##NAME##Attr(ir);              \
      std::optional<ETY1> v1 = get##ENAME1##From##NAME##Attr(ir);              \
      std::optional<ETY2> v2 = get##ENAME2##From##NAME##Attr(ir);              \
                                                                               \
      isValid &= VERIFY_IMPL(v0, os, ETY0, ENAME0, EN0, IRNAME);               \
      isValid &= VERIFY_IMPL(v1, os, ETY1, ENAME1, EN1, IRNAME);               \
      isValid &= VERIFY_IMPL(v2, os, ETY2, ENAME2, EN2, IRNAME);               \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        isValid &= verify##NAME##Attr(ir, *v0, *v1, *v2, os);                  \
    }                                                                          \
    return isValid;                                                            \
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
  bool llvm::verify##NAME##Attr(const IRELEM &ir, raw_ostream *os) {           \
    bool isValid = true;                                                       \
    if (has##NAME##Attr(ir)) {                                                 \
      std::optional<ETY0> v0 = get##ENAME0##From##NAME##Attr(ir);              \
      std::optional<ETY1> v1 = get##ENAME1##From##NAME##Attr(ir);              \
      std::optional<ETY2> v2 = get##ENAME2##From##NAME##Attr(ir);              \
      std::optional<ETY3> v3 = get##ENAME3##From##NAME##Attr(ir);              \
                                                                               \
      isValid &= VERIFY_IMPL(v0, os, ETY0, ENAME0, EN0, IRNAME);               \
      isValid &= VERIFY_IMPL(v1, os, ETY1, ENAME1, EN1, IRNAME);               \
      isValid &= VERIFY_IMPL(v2, os, ETY2, ENAME2, EN2, IRNAME);               \
      isValid &= VERIFY_IMPL(v3, os, ETY3, ENAME3, EN3, IRNAME);               \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        isValid &= verify##NAME##Attr(ir, *v0, *v1, *v2, *v3, os);             \
    }                                                                          \
    return isValid;                                                            \
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
  bool llvm::verify##NAME##Attr(const IRELEM &ir, raw_ostream *os) {           \
    bool isValid = true;                                                       \
    if (has##NAME##Attr(ir)) {                                                 \
      std::optional<ETY0> v0 = get##ENAME0##From##NAME##Attr(ir);              \
      std::optional<ETY1> v1 = get##ENAME1##From##NAME##Attr(ir);              \
      std::optional<ETY2> v2 = get##ENAME2##From##NAME##Attr(ir);              \
      std::optional<ETY3> v3 = get##ENAME3##From##NAME##Attr(ir);              \
      std::optional<ETY4> v4 = get##ENAME4##From##NAME##Attr(ir);              \
                                                                               \
      isValid &= VERIFY_IMPL(v0, os, ETY0, ENAME0, EN0, IRNAME);               \
      isValid &= VERIFY_IMPL(v1, os, ETY1, ENAME1, EN1, IRNAME);               \
      isValid &= VERIFY_IMPL(v2, os, ETY2, ENAME2, EN2, IRNAME);               \
      isValid &= VERIFY_IMPL(v3, os, ETY3, ENAME3, EN3, IRNAME);               \
      isValid &= VERIFY_IMPL(v4, os, ETY4, ENAME4, EN4, IRNAME);               \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        isValid &= verify##NAME##Attr(ir, *v0, *v1, *v2, *v3, *v4, os);        \
    }                                                                          \
    return isValid;                                                            \
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
  bool llvm::verify##NAME##Attr(const IRELEM &ir, raw_ostream *os) {           \
    bool isValid = true;                                                       \
    if (has##NAME##Attr(ir)) {                                                 \
      std::optional<ETY0> v0 = get##ENAME0##From##NAME##Attr(ir);              \
      std::optional<ETY1> v1 = get##ENAME1##From##NAME##Attr(ir);              \
      std::optional<ETY2> v2 = get##ENAME2##From##NAME##Attr(ir);              \
      std::optional<ETY3> v3 = get##ENAME3##From##NAME##Attr(ir);              \
      std::optional<ETY4> v4 = get##ENAME4##From##NAME##Attr(ir);              \
      std::optional<ETY5> v5 = get##ENAME5##From##NAME##Attr(ir);              \
                                                                               \
      isValid &= VERIFY_IMPL(v0, os, ETY0, ENAME0, EN0, IRNAME);               \
      isValid &= VERIFY_IMPL(v1, os, ETY1, ENAME1, EN1, IRNAME);               \
      isValid &= VERIFY_IMPL(v2, os, ETY2, ENAME2, EN2, IRNAME);               \
      isValid &= VERIFY_IMPL(v3, os, ETY3, ENAME3, EN3, IRNAME);               \
      isValid &= VERIFY_IMPL(v4, os, ETY4, ENAME4, EN4, IRNAME);               \
      isValid &= VERIFY_IMPL(v5, os, ETY5, ENAME5, EN5, IRNAME);               \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        isValid &= verify##NAME##Attr(ir, *v0, *v1, *v2, *v3, *v4, *v5, os);   \
    }                                                                          \
    return isValid;                                                            \
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
  bool llvm::verify##NAME##Attr(const IRELEM &ir, raw_ostream *os) {           \
    bool isValid = true;                                                       \
    if (has##NAME##Attr(ir)) {                                                 \
      std::optional<ETY0> v0 = get##ENAME0##From##NAME##Attr(ir);              \
      std::optional<ETY1> v1 = get##ENAME1##From##NAME##Attr(ir);              \
      std::optional<ETY2> v2 = get##ENAME2##From##NAME##Attr(ir);              \
      std::optional<ETY3> v3 = get##ENAME3##From##NAME##Attr(ir);              \
      std::optional<ETY4> v4 = get##ENAME4##From##NAME##Attr(ir);              \
      std::optional<ETY5> v5 = get##ENAME5##From##NAME##Attr(ir);              \
      std::optional<ETY6> v6 = get##ENAME6##From##NAME##Attr(ir);              \
                                                                               \
      isValid &= VERIFY_IMPL(v0, os, ETY0, ENAME0, EN0, IRNAME);               \
      isValid &= VERIFY_IMPL(v1, os, ETY1, ENAME1, EN1, IRNAME);               \
      isValid &= VERIFY_IMPL(v2, os, ETY2, ENAME2, EN2, IRNAME);               \
      isValid &= VERIFY_IMPL(v3, os, ETY3, ENAME3, EN3, IRNAME);               \
      isValid &= VERIFY_IMPL(v4, os, ETY4, ENAME4, EN4, IRNAME);               \
      isValid &= VERIFY_IMPL(v5, os, ETY5, ENAME5, EN5, IRNAME);               \
      isValid &= VERIFY_IMPL(v6, os, ETY6, ENAME6, EN6, IRNAME);               \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        isValid &=                                                             \
            verify##NAME##Attr(ir, *v0, *v1, *v2, *v3, *v4, *v5, *v6, os);     \
    }                                                                          \
    return isValid;                                                            \
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
  bool llvm::verify##NAME##Attr(const IRELEM &ir, raw_ostream *os) {           \
    bool isValid = true;                                                       \
    if (has##NAME##Attr(ir)) {                                                 \
      std::optional<ETY0> v0 = get##ENAME0##From##NAME##Attr(ir);              \
      std::optional<ETY1> v1 = get##ENAME1##From##NAME##Attr(ir);              \
      std::optional<ETY2> v2 = get##ENAME2##From##NAME##Attr(ir);              \
      std::optional<ETY3> v3 = get##ENAME3##From##NAME##Attr(ir);              \
      std::optional<ETY4> v4 = get##ENAME4##From##NAME##Attr(ir);              \
      std::optional<ETY5> v5 = get##ENAME5##From##NAME##Attr(ir);              \
      std::optional<ETY6> v6 = get##ENAME6##From##NAME##Attr(ir);              \
      std::optional<ETY7> v7 = get##ENAME7##From##NAME##Attr(ir);              \
                                                                               \
      isValid &= VERIFY_IMPL(v0, os, ETY0, ENAME0, EN0, IRNAME);               \
      isValid &= VERIFY_IMPL(v1, os, ETY1, ENAME1, EN1, IRNAME);               \
      isValid &= VERIFY_IMPL(v2, os, ETY2, ENAME2, EN2, IRNAME);               \
      isValid &= VERIFY_IMPL(v3, os, ETY3, ENAME3, EN3, IRNAME);               \
      isValid &= VERIFY_IMPL(v4, os, ETY4, ENAME4, EN4, IRNAME);               \
      isValid &= VERIFY_IMPL(v5, os, ETY5, ENAME5, EN5, IRNAME);               \
      isValid &= VERIFY_IMPL(v6, os, ETY6, ENAME6, EN6, IRNAME);               \
      isValid &= VERIFY_IMPL(v7, os, ETY7, ENAME7, EN7, IRNAME);               \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        isValid &= verify##NAME##Attr(ir, *v0, *v1, *v2, *v3, *v4, *v5, *v6,   \
                                      *v7, os);                                \
    }                                                                          \
    return isValid;                                                            \
  }

#define DEFN_ATTR_N(IRELEM, NAME, IRNAME, ETY, ENAME, EN, NELEMS)              \
  std::optional<ETY> llvm::get##ENAME##From##NAME##Attr(const IRELEM &ir) {    \
    return getAttrValue<ETY>(IRNAME, getAttrList(ir), EN, NELEMS);             \
  }

#endif // KITSUNE_LIB_CORE_ATTRS_IMPL_H

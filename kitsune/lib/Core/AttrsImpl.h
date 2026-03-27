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

#include "kitsune/Core/AttrsCommon.h"
#include "kitsune/Support/ToString.h"
#include "llvm/Analysis/LoopInfo.h"

#define VERIFY_IMPL(OS, OBJ, V, NAME, IRNAME, ETYPE, ENAME, EN)                \
  do {                                                                         \
    if (!V.has_value()) {                                                      \
      if (OS)                                                                  \
        (*OS) << "Missing value of type '" << toString<ETYPE>()                \
              << "' for element '" << #ENAME << "' at index '" << EN           \
              << "' in attribute '" << IRNAME << "'\n";                        \
      return false;                                                            \
    }                                                                          \
  } while (0)

#define DEFN_ATTR_GENERIC(IRTYPE, ENUMKIND)                                    \
  static void setAttrList(IRTYPE &, MDNode *attrList);                         \
  static void addAttr(IRTYPE &, StringRef name, ArrayRef<Metadata *> vals);    \
  static void removeAttr(IRTYPE &, StringRef attrName);                        \
                                                                               \
  bool llvm::hasAttr(const IRTYPE &ir, ENUMKIND attr) {                        \
    return getRawAttr(getAttrName(attr), getAttrList(ir));                     \
  }                                                                            \
                                                                               \
  void llvm::removeAttr(IRTYPE &ir, ENUMKIND attr) {                           \
    ::removeAttr(ir, getAttrName(attr));                                       \
  }

#define DEFN_ATTR_COMMON(IRTYPE, ENUMKIND, NAME, IRNAME, TYPE)                 \
  bool llvm::has##NAME##Attr(const IRTYPE &ir) {                               \
    return getRawAttr(IRNAME, getAttrList(ir));                                \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(IRTYPE &ir) { ::removeAttr(ir, IRNAME); }

#define DEFN_ATTR_LOOP(IRTYPE, NAME, IRNAME)                                   \
  std::optional<Loop *> llvm::get##NAME##Attr(                                 \
      const IRTYPE &ir, const SmallVectorImpl<const LoopInfo *> &lis) {        \
    return getAttrValue(IRNAME, getAttrList(ir), lis);                         \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(IRTYPE &ir, const Loop &loop) {                   \
    ::addAttr(ir, IRNAME, loop.getLoopID());                                   \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const IRTYPE &ir, raw_ostream *os) {           \
    return verifyAttrLoop(IRNAME, getAttrList(ir), os);                        \
  }

#define DEFN_ATTR_0(IRTYPE, NAME, IRNAME)                                      \
  void llvm::add##NAME##Attr(IRTYPE &ir) { ::addAttr(ir, IRNAME, {}); }        \
                                                                               \
  bool llvm::verify##NAME##Attr(const IRTYPE &ir, raw_ostream *os) {           \
    return verifyAttr0(IRNAME, getAttrList(ir), os);                           \
  }

#define DEFN_ATTR_1(IRTYPE, NAME, IRNAME, TYPE)                                \
  std::optional<TYPE> llvm::get##NAME##Attr(const IRTYPE &ir) {                \
    return getAttrValue<TYPE>(IRNAME, getAttrList(ir), 0, 1);                  \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(IRTYPE &ir, const TYPE &val) {                    \
    LLVMContext &ctx = getContext(ir);                                         \
    Metadata *attrVals[] = {toMetadata(val, ctx)};                             \
    ::addAttr(ir, IRNAME, attrVals);                                           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const IRTYPE &ir, raw_ostream *os) {           \
    if (has##NAME##Attr(ir)) {                                                 \
      std::optional<TYPE> v = get##NAME##Attr(ir);                             \
      if (!v.has_value()) {                                                    \
        if (os)                                                                \
          (*os) << "Missing value of type '" << toString<TYPE>()               \
                << "' in attribute '" << IRNAME << "'\n";                      \
        return false;                                                          \
      }                                                                        \
    }                                                                          \
    return true;                                                               \
  }

#define DEFN_ATTR_2(IRTYPE, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1,     \
                    EN1)                                                       \
  void llvm::add##NAME##Attr(IRTYPE &ir, const ETY0 &e0, const ETY1 &e1) {     \
    LLVMContext &ctx = getContext(ir);                                         \
    Metadata *attrVals[] = {toMetadata(e0, ctx), toMetadata(e1, ctx)};         \
    ::addAttr(ir, IRNAME, attrVals);                                           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const IRTYPE &ir, raw_ostream *os) {           \
    if (has##NAME##Attr(ir)) {                                                 \
      std::optional<ETY0> v0 = get##ENAME0##From##NAME##Attr(ir);              \
      std::optional<ETY1> v1 = get##ENAME1##From##NAME##Attr(ir);              \
                                                                               \
      VERIFY_IMPL(os, ir, v0, NAME, IRNAME, ETY0, ENAME0, EN0);                \
      VERIFY_IMPL(os, ir, v1, NAME, IRNAME, ETY1, ENAME1, EN1);                \
    }                                                                          \
    return true;                                                               \
  }

#define DEFN_ATTR_3(IRTYPE, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1,     \
                    EN1, ETY2, ENAME2, EN2)                                    \
  void llvm::add##NAME##Attr(IRTYPE &ir, const ETY0 &e0, const ETY1 &e1,       \
                             const ETY2 &e2) {                                 \
    LLVMContext &ctx = getContext(ir);                                         \
    Metadata *attrVals[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),          \
                            toMetadata(e2, ctx)};                              \
    ::addAttr(ir, IRNAME, attrVals);                                           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const IRTYPE &ir, raw_ostream *os) {           \
    if (has##NAME##Attr(ir)) {                                                 \
      std::optional<ETY0> v0 = get##ENAME0##From##NAME##Attr(ir);              \
      std::optional<ETY1> v1 = get##ENAME1##From##NAME##Attr(ir);              \
      std::optional<ETY2> v2 = get##ENAME2##From##NAME##Attr(ir);              \
                                                                               \
      VERIFY_IMPL(os, ir, v0, NAME, IRNAME, ETY0, ENAME0, EN0);                \
      VERIFY_IMPL(os, ir, v1, NAME, IRNAME, ETY1, ENAME1, EN1);                \
      VERIFY_IMPL(os, ir, v2, NAME, IRNAME, ETY2, ENAME2, EN2);                \
    }                                                                          \
    return true;                                                               \
  }

#define DEFN_ATTR_4(IRTYPE, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1,     \
                    EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3)                 \
  void llvm::add##NAME##Attr(IRTYPE &ir, const ETY0 &e0, const ETY1 &e1,       \
                             const ETY2 &e2, const ETY3 &e3) {                 \
    LLVMContext &ctx = getContext(ir);                                         \
    Metadata *attrVals[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),          \
                            toMetadata(e2, ctx), toMetadata(e3, ctx)};         \
    ::addAttr(ir, IRNAME, attrVals);                                           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const IRTYPE &ir, raw_ostream *os) {           \
    if (has##NAME##Attr(ir)) {                                                 \
      std::optional<ETY0> v0 = get##ENAME0##From##NAME##Attr(ir);              \
      std::optional<ETY1> v1 = get##ENAME1##From##NAME##Attr(ir);              \
      std::optional<ETY2> v2 = get##ENAME2##From##NAME##Attr(ir);              \
      std::optional<ETY3> v3 = get##ENAME3##From##NAME##Attr(ir);              \
                                                                               \
      VERIFY_IMPL(os, ir, v0, NAME, IRNAME, ETY0, ENAME0, EN0);                \
      VERIFY_IMPL(os, ir, v1, NAME, IRNAME, ETY1, ENAME1, EN1);                \
      VERIFY_IMPL(os, ir, v2, NAME, IRNAME, ETY2, ENAME2, EN2);                \
      VERIFY_IMPL(os, ir, v3, NAME, IRNAME, ETY3, ENAME3, EN3);                \
    }                                                                          \
    return true;                                                               \
  }

#define DEFN_ATTR_5(IRTYPE, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1,     \
                    EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4,   \
                    EN4)                                                       \
  void llvm::add##NAME##Attr(IRTYPE &ir, const ETY0 &e0, const ETY1 &e1,       \
                             const ETY2 &e2, const ETY3 &e3, const ETY4 &e4) { \
    LLVMContext &ctx = getContext(ir);                                         \
    Metadata *attrVals[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),          \
                            toMetadata(e2, ctx), toMetadata(e3, ctx),          \
                            toMetadata(e4, ctx)};                              \
    ::addAttr(ir, IRNAME, attrVals);                                           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const IRTYPE &ir, raw_ostream *os) {           \
    if (has##NAME##Attr(ir)) {                                                 \
      std::optional<ETY0> v0 = get##ENAME0##From##NAME##Attr(ir);              \
      std::optional<ETY1> v1 = get##ENAME1##From##NAME##Attr(ir);              \
      std::optional<ETY2> v2 = get##ENAME2##From##NAME##Attr(ir);              \
      std::optional<ETY3> v3 = get##ENAME3##From##NAME##Attr(ir);              \
      std::optional<ETY4> v4 = get##ENAME4##From##NAME##Attr(ir);              \
                                                                               \
      VERIFY_IMPL(os, ir, v0, NAME, IRNAME, ETY0, ENAME0, EN0);                \
      VERIFY_IMPL(os, ir, v1, NAME, IRNAME, ETY1, ENAME1, EN1);                \
      VERIFY_IMPL(os, ir, v2, NAME, IRNAME, ETY2, ENAME2, EN2);                \
      VERIFY_IMPL(os, ir, v3, NAME, IRNAME, ETY3, ENAME3, EN3);                \
      VERIFY_IMPL(os, ir, v4, NAME, IRNAME, ETY4, ENAME4, EN4);                \
    }                                                                          \
    return true;                                                               \
  }

#define DEFN_ATTR_6(IRTYPE, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1,     \
                    EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4,   \
                    EN4, ETY5, ENAME5, EN5)                                    \
  void llvm::add##NAME##Attr(IRTYPE &ir, const ETY0 &e0, const ETY1 &e1,       \
                             const ETY2 &e2, const ETY3 &e3, const ETY4 &e4,   \
                             const ETY5 &e5) {                                 \
    LLVMContext &ctx = getContext(ir);                                         \
    Metadata *attrVals[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),          \
                            toMetadata(e2, ctx), toMetadata(e3, ctx),          \
                            toMetadata(e4, ctx), toMetadata(e5, ctx)};         \
    ::addAttr(ir, IRNAME, attrVals);                                           \
  }                                                                            \
                                                                               \
  bool llvm::verify##NAME##Attr(const IRTYPE &ir, raw_ostream *os) {           \
    if (has##NAME##Attr(ir)) {                                                 \
      std::optional<ETY0> v0 = get##ENAME0##From##NAME##Attr(ir);              \
      std::optional<ETY1> v1 = get##ENAME1##From##NAME##Attr(ir);              \
      std::optional<ETY2> v2 = get##ENAME2##From##NAME##Attr(ir);              \
      std::optional<ETY3> v3 = get##ENAME3##From##NAME##Attr(ir);              \
      std::optional<ETY4> v4 = get##ENAME4##From##NAME##Attr(ir);              \
      std::optional<ETY5> v5 = get##ENAME5##From##NAME##Attr(ir);              \
                                                                               \
      VERIFY_IMPL(os, ir, v0, NAME, IRNAME, ETY0, ENAME0, EN0);                \
      VERIFY_IMPL(os, ir, v1, NAME, IRNAME, ETY1, ENAME1, EN1);                \
      VERIFY_IMPL(os, ir, v2, NAME, IRNAME, ETY2, ENAME2, EN2);                \
      VERIFY_IMPL(os, ir, v3, NAME, IRNAME, ETY3, ENAME3, EN3);                \
      VERIFY_IMPL(os, ir, v4, NAME, IRNAME, ETY4, ENAME4, EN4);                \
      VERIFY_IMPL(os, ir, v5, NAME, IRNAME, ETY5, ENAME5, EN5);                \
    }                                                                          \
    return true;                                                               \
  }

#define DEFN_ATTR_7(IRTYPE, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1,     \
                    EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4,   \
                    EN4, ETY5, ENAME5, EN5, ETY6, ENAME6, EN6)                 \
  void llvm::add##NAME##Attr(IRTYPE &ir, const ETY0 &e0, const ETY1 &e1,       \
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
  bool llvm::verify##NAME##Attr(const IRTYPE &ir, raw_ostream *os) {           \
    if (has##NAME##Attr(ir)) {                                                 \
      std::optional<ETY0> v0 = get##ENAME0##From##NAME##Attr(ir);              \
      std::optional<ETY1> v1 = get##ENAME1##From##NAME##Attr(ir);              \
      std::optional<ETY2> v2 = get##ENAME2##From##NAME##Attr(ir);              \
      std::optional<ETY3> v3 = get##ENAME3##From##NAME##Attr(ir);              \
      std::optional<ETY4> v4 = get##ENAME4##From##NAME##Attr(ir);              \
      std::optional<ETY5> v5 = get##ENAME5##From##NAME##Attr(ir);              \
      std::optional<ETY6> v6 = get##ENAME6##From##NAME##Attr(ir);              \
                                                                               \
      VERIFY_IMPL(os, ir, v0, NAME, IRNAME, ETY0, ENAME0, EN0);                \
      VERIFY_IMPL(os, ir, v1, NAME, IRNAME, ETY1, ENAME1, EN1);                \
      VERIFY_IMPL(os, ir, v2, NAME, IRNAME, ETY2, ENAME2, EN2);                \
      VERIFY_IMPL(os, ir, v3, NAME, IRNAME, ETY3, ENAME3, EN3);                \
      VERIFY_IMPL(os, ir, v4, NAME, IRNAME, ETY4, ENAME4, EN4);                \
      VERIFY_IMPL(os, ir, v5, NAME, IRNAME, ETY5, ENAME5, EN5);                \
      VERIFY_IMPL(os, ir, v6, NAME, IRNAME, ETY6, ENAME6, EN6);                \
    }                                                                          \
    return true;                                                               \
  }

#define DEFN_ATTR_8(IRTYPE, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1,     \
                    EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4,   \
                    EN4, ETY5, ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7,   \
                    EN7)                                                       \
  void llvm::add##NAME##Attr(IRTYPE &ir, const ETY0 &e0, const ETY1 &e1,       \
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
  bool llvm::verify##NAME##Attr(const IRTYPE &ir, raw_ostream *os) {           \
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
      VERIFY_IMPL(os, ir, v0, NAME, IRNAME, ETY0, ENAME0, EN0);                \
      VERIFY_IMPL(os, ir, v1, NAME, IRNAME, ETY1, ENAME1, EN1);                \
      VERIFY_IMPL(os, ir, v2, NAME, IRNAME, ETY2, ENAME2, EN2);                \
      VERIFY_IMPL(os, ir, v3, NAME, IRNAME, ETY3, ENAME3, EN3);                \
      VERIFY_IMPL(os, ir, v4, NAME, IRNAME, ETY4, ENAME4, EN4);                \
      VERIFY_IMPL(os, ir, v5, NAME, IRNAME, ETY5, ENAME5, EN5);                \
      VERIFY_IMPL(os, ir, v6, NAME, IRNAME, ETY6, ENAME6, EN6);                \
      VERIFY_IMPL(os, ir, v7, NAME, IRNAME, ETY7, ENAME7, EN7);                \
    }                                                                          \
    return true;                                                               \
  }

#define DEFN_ATTR_N(IRTYPE, NAME, IRNAME, ETY, ENAME, EN, NELEMS)              \
  std::optional<ETY> llvm::get##ENAME##From##NAME##Attr(const IRTYPE &ir) {    \
    return getAttrValue<ETY>(IRNAME, getAttrList(ir), EN, NELEMS);             \
  }

#endif // KITSUNE_LIB_CORE_ATTRS_IMPL_H

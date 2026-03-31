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

#include "kitsune/Core/MetadataUtils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LoopInfo.h"

namespace llvm {

class LLVMContext;
class MDNode;

class KitVerifier;

namespace detail {

/// Create a new empty attribute list. This will be of the form
///
/// \code{llvm}
///     !0 = distinct !{!0}
/// \endcode
MDNode *getNewAttrList(LLVMContext &ctx);

/// Get an attribute list containing the attribute with name \p attrName and
/// values \p attrVals. \p attrList is the existing attribute list. It may be
/// null in which case the returned list will contain a single attribute. If
/// the attribute already exists in \p attrList, its value(s) will be replaced
/// with new new value(s).
///
/// Some examples are provided below. In each, a call is followed by an example
/// of the new attribute list that will be returned. An optional old attribute
/// list may also be provided.
///
/// **New attribute list**
///
/// \code{c++}
///     getAttrListWithAttr("new-attr", {...}, nullptr, ctx);
/// \endcode
///
/// This will return the following new attribute list.
///
/// \code{llvm}
///     !0 = distinct !{!0, !1}
///     !1 = !{!"new-attr", ...}
/// \endcode{llvm}
///
///
/// **Add an attribute that is not in the list**
///
/// \code{llvm}
///     !0 = distinct !{!0, !1, !2}
///     !1 = !{!"attr-flag"}
///     !2 = !{!"attr-1", i32 32767}
/// \endcode
///
/// \code{c++}
///     getAttrListWithAttr("new-attr", {...}, <attrList>, ctx);
/// \endcode
///
/// \code{c++}
///     !0 = distinct !{!0, !1, !2, !3}
///     !1 = !{!"attr-flag"}
///     !2 = !{!"attr-1", i32 32767}
///     !3 = !{!"new-attr", ...}
/// \endcode
///
///
/// **Update the value of an attribute in the list**
///
/// \code{llvm}
///     !0 = distinct !{!0, !1, !2}
///     !1 = !{!"new-attr", !"old"}
///     !2 = !{!"attr-flag"}
/// \endcode
///
/// \code{c++}
///     getAttrListWithAttr("new-attr", {!"new"}, <attrList>, ctx);
/// \endcode
///
/// \code{llvm}
///     !0 = distinct !{!0, !1, !2}
///     !1 = !{!"attr-flag"}
///     !2 = !{!"new-attr", !"new"}
/// \endcode
///
MDNode *getAttrListWith(StringRef attrName, const ArrayRef<Metadata *> attrVals,
                        MDNode *attrList, LLVMContext &ctx);

/// Remove the attribute named \p attrName from \p attrList. If the attribute
/// exists in the list, a new MDNode will be created and returned. Otherwise,
/// \p attrList will be returned. If removing the result would result in an
/// empty list, return nullptr. If \p attrList is nullptr, returns nullptr.
MDNode *getAttrListWithout(StringRef attrName, MDNode *attrList);

/// Create a raw attribute metadata node with name \p attrName and values
/// \p attrVals. This will be of the form
///
/// \code{llvm}
///     !0 = !{!"<NAME>", ...}
/// \endcode
///
/// where <NAME> is the name of the attribute as specified in \p attrName and
/// the ellipses denote the metadata in \p attrVals.
MDNode *makeRawAttr(LLVMContext &ctx, StringRef attrName,
                    ArrayRef<Metadata *> vals);

/// If the attribute list \p attrList contains an attribute \p attrName, return
/// the MDNode for that attribute. Otherwise, return nullptr. If found, the
/// MDNode that is returned will have at least one operand. This will be an
/// MDString whose value is the name of the attribute. If any other operands
/// are present, they will be the values accepted by the attribute. If
/// \p attrList is nullptr, this will also return nullptr.
MDNode *getRawAttr(StringRef attrName, const MDNode *attrList);

/// Get the name of the attribute \p attr.
StringRef getRawAttrName(const MDNode &attr);

/// Get the value of the raw attribute that is expected to have a exactly one
/// value that is an LLVM Loop.
std::optional<Loop *>
getRawAttrValue(const MDNode &attr,
                const SmallVectorImpl<const LoopInfo *> &lis);

/// Get the value of the \p i'th value from the raw attribute \p attr that is
/// expected to be of type \p T. If the value is not present, or if it is not of
/// type \p T, return std::nullopt.
template <typename T>
std::optional<T> getRawAttrValue(const MDNode &attr, unsigned i);

/// Verify that the raw attribute \p attr has the expected number of values,
/// \p attrVals. If so, return true. Otherwise, if an optional output stream,
/// \p os, has been provided, write an error message to it.
bool verifyRawAttrValueCount(KitVerifier &v, const MDNode &attr,
                             unsigned attrVals);

/// Verify an attribute \p attr that is expected to have a single value. This
/// value is an MDNode that corresponds to the ID of a loop. Return false if
/// \p attrName is present in \p attrList and does not have exactly one value.
/// Without a LoopInfo object, it is impossible to truly verify that the value
/// is the ID of a loop. Instead, some rudimentary checks are performed - in
/// particular that the MDNode is distinct and the first operand is a
/// self-reference. If any of these is not the case, return false. Return true
/// in all other cases, If false is due to be returned, and the optional output
/// stream \p os is not nullptr, print an error message to it.
bool verifyRawAttrValueLoop(KitVerifier &v, const MDNode &attr);

/// Verify that a raw attribute \p attr has a value of type \p T at index \p i.
/// \p i must be in the range [0, N) where N is the number of values that the
/// attribute expects.
template <typename T>
bool verifyRawAttrValueAt(KitVerifier &v, const MDNode &attr, unsigned i,
                          const std::optional<T> &val);

template <typename T, typename... Vals>
bool verifyRawAttrValuesImpl(KitVerifier &v, const MDNode &attr, unsigned i,
                             const std::optional<T> &val, const Vals &...vals) {
  bool ok = verifyRawAttrValueAt(v, attr, i, val);
  if constexpr (sizeof...(Vals))
    ok &= detail::verifyRawAttrValuesImpl(v, attr, i + 1, vals...);
  return ok;
}

/// Check that the std::optional values, \p vals. If all of them have values,
/// return true. Otherwise, return false and write an error to the \p os if it
/// is not nullptr.
template <typename... Vals>
bool verifyRawAttrValues(KitVerifier &v, const MDNode &attr,
                         const Vals &...vals) {
  return detail::verifyRawAttrValuesImpl(v, attr, 0, vals...);
}

} // namespace detail

} // namespace llvm

#define DEFN_ATTR_GENERIC(IRELEM, KIND)                                        \
  bool llvm::hasAttr(const IRELEM &ir, KIND attr) {                            \
    return detail::getRawAttr(getAttrName(attr), detail::getRawAttrList(ir));  \
  }                                                                            \
                                                                               \
  void llvm::removeAttr(IRELEM &ir, KIND attr) {                               \
    detail::removeAttr(ir, getAttrName(attr));                                 \
  }                                                                            \
                                                                               \
  iterator_range<detail::AttrIterator> llvm::detail::attrs(const IRELEM &ir) { \
    if (const MDNode *attrList = detail::getRawAttrList(ir)) {                 \
      detail::AttrIterator beg(attrList);                                      \
      detail::AttrIterator end(attrList, attrList->getNumOperands());          \
                                                                               \
      return iterator_range(beg, end);                                         \
    }                                                                          \
    return iterator_range(AttrIterator(), AttrIterator());                     \
  }

#define DEFN_ATTR_COMMON(IRELEM, KIND, NAME, IRNAME, CUSTOMVERIFY, TYPE)       \
  bool llvm::has##NAME##Attr(const IRELEM &ir) {                               \
    return detail::getRawAttr(IRNAME, detail::getRawAttrList(ir));             \
  }                                                                            \
                                                                               \
  void llvm::remove##NAME##Attr(IRELEM &ir) { detail::removeAttr(ir, IRNAME); }

#define DEFN_ATTR_LOOP(IRELEM, NAME, IRNAME, CUSTOMVERIFY)                     \
  std::optional<Loop *> llvm::get##NAME##Attr(                                 \
      const IRELEM &ir, const SmallVectorImpl<const LoopInfo *> &lis) {        \
    if (const MDNode *attr =                                                   \
            detail::getRawAttr(IRNAME, detail::getRawAttrList(ir)))            \
      return detail::getRawAttrValue(*attr, lis);                              \
    return std::nullopt;                                                       \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(IRELEM &ir, const Loop &loop) {                   \
    detail::addAttr(ir, IRNAME, loop.getLoopID());                             \
  }                                                                            \
                                                                               \
  void llvm::verify##NAME##Attr(KitVerifier &v, const IRELEM &ir) {            \
    if (const MDNode *attr =                                                   \
            detail::getRawAttr(IRNAME, detail::getRawAttrList(ir))) {          \
      if (!detail::verifyRawAttrValueCount(v, *attr, 1))                       \
        return;                                                                \
                                                                               \
      if (!detail::verifyRawAttrValueLoop(v, *attr))                           \
        return;                                                                \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        verify##NAME##Attr(v, ir, *cast<MDNode>(attr->getOperand(1)));         \
    }                                                                          \
  }

#define DEFN_ATTR_0(IRELEM, NAME, IRNAME, CUSTOMVERIFY)                        \
  void llvm::add##NAME##Attr(IRELEM &ir) { detail::addAttr(ir, IRNAME, {}); }  \
                                                                               \
  void llvm::verify##NAME##Attr(KitVerifier &v, const IRELEM &ir) {            \
    if (const MDNode *attr =                                                   \
            detail::getRawAttr(IRNAME, detail::getRawAttrList(ir))) {          \
      if (!detail::verifyRawAttrValueCount(v, *attr, 0))                       \
        return;                                                                \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        verify##NAME##Attr(v, ir, true);                                       \
    }                                                                          \
  }

#define DEFN_ATTR_1(IRELEM, NAME, IRNAME, CUSTOMVERIFY, TYPE)                  \
  std::optional<TYPE> llvm::get##NAME##Attr(const IRELEM &ir) {                \
    if (const MDNode *attr =                                                   \
            detail::getRawAttr(IRNAME, detail::getRawAttrList(ir)))            \
      if (attr->getNumOperands() == 2)                                         \
        return detail::getRawAttrValue<TYPE>(*attr, 0);                        \
    return std::nullopt;                                                       \
  }                                                                            \
                                                                               \
  void llvm::add##NAME##Attr(IRELEM &ir, const TYPE &val) {                    \
    LLVMContext &ctx = getContext(ir);                                         \
    Metadata *attrVals[] = {toMetadata(val, ctx)};                             \
    detail::addAttr(ir, IRNAME, attrVals);                                     \
  }                                                                            \
                                                                               \
  void llvm::verify##NAME##Attr(KitVerifier &v, const IRELEM &ir) {            \
    if (const MDNode *attr =                                                   \
            detail::getRawAttr(IRNAME, detail::getRawAttrList(ir))) {          \
      if (!detail::verifyRawAttrValueCount(v, *attr, 1))                       \
        return;                                                                \
                                                                               \
      std::optional<TYPE> val = detail::getRawAttrValue<TYPE>(*attr, 0);       \
      if (!detail::verifyRawAttrValues(v, *attr, val))                         \
        return;                                                                \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        verify##NAME##Attr(v, ir, *val);                                       \
    }                                                                          \
  }

#define DEFN_ATTR_2(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1)                                         \
  void llvm::add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1) {     \
    LLVMContext &ctx = getContext(ir);                                         \
    Metadata *attrVals[] = {toMetadata(e0, ctx), toMetadata(e1, ctx)};         \
    detail::addAttr(ir, IRNAME, attrVals);                                     \
  }                                                                            \
                                                                               \
  void llvm::verify##NAME##Attr(KitVerifier &v, const IRELEM &ir) {            \
    if (const MDNode *attr =                                                   \
            detail::getRawAttr(IRNAME, detail::getRawAttrList(ir))) {          \
      if (!detail::verifyRawAttrValueCount(v, *attr, 2))                       \
        return;                                                                \
                                                                               \
      std::optional<ETY0> v0 = detail::getRawAttrValue<ETY0>(*attr, 0);        \
      std::optional<ETY1> v1 = detail::getRawAttrValue<ETY1>(*attr, 1);        \
      if (!detail::verifyRawAttrValues(v, *attr, v0, v1))                      \
        return;                                                                \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        verify##NAME##Attr(v, ir, *v0, *v1);                                   \
    }                                                                          \
  }

#define DEFN_ATTR_3(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2)                      \
  void llvm::add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,       \
                             const ETY2 &e2) {                                 \
    LLVMContext &ctx = getContext(ir);                                         \
    Metadata *attrVals[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),          \
                            toMetadata(e2, ctx)};                              \
    detail::addAttr(ir, IRNAME, attrVals);                                     \
  }                                                                            \
                                                                               \
  void llvm::verify##NAME##Attr(KitVerifier &v, const IRELEM &ir) {            \
    if (const MDNode *attr =                                                   \
            detail::getRawAttr(IRNAME, detail::getRawAttrList(ir))) {          \
      if (!detail::verifyRawAttrValueCount(v, *attr, 3))                       \
        return;                                                                \
                                                                               \
      std::optional<ETY0> v0 = detail::getRawAttrValue<ETY0>(*attr, 0);        \
      std::optional<ETY1> v1 = detail::getRawAttrValue<ETY1>(*attr, 1);        \
      std::optional<ETY2> v2 = detail::getRawAttrValue<ETY2>(*attr, 2);        \
      if (!detail::verifyRawAttrValues(v, *attr, v0, v1, v2))                  \
        return;                                                                \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        verify##NAME##Attr(v, ir, *v0, *v1, *v2);                              \
    }                                                                          \
  }

#define DEFN_ATTR_4(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3)   \
  void llvm::add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,       \
                             const ETY2 &e2, const ETY3 &e3) {                 \
    LLVMContext &ctx = getContext(ir);                                         \
    Metadata *attrVals[] = {toMetadata(e0, ctx), toMetadata(e1, ctx),          \
                            toMetadata(e2, ctx), toMetadata(e3, ctx)};         \
    detail::addAttr(ir, IRNAME, attrVals);                                     \
  }                                                                            \
                                                                               \
  void llvm::verify##NAME##Attr(KitVerifier &v, const IRELEM &ir) {            \
    if (const MDNode *attr =                                                   \
            detail::getRawAttr(IRNAME, detail::getRawAttrList(ir))) {          \
      if (!detail::verifyRawAttrValueCount(v, *attr, 4))                       \
        return;                                                                \
                                                                               \
      std::optional<ETY0> v0 = detail::getRawAttrValue<ETY0>(*attr, 0);        \
      std::optional<ETY1> v1 = detail::getRawAttrValue<ETY1>(*attr, 1);        \
      std::optional<ETY2> v2 = detail::getRawAttrValue<ETY2>(*attr, 2);        \
      std::optional<ETY3> v3 = detail::getRawAttrValue<ETY3>(*attr, 3);        \
      if (!detail::verifyRawAttrValues(v, *attr, v0, v1, v2, v3))              \
        return;                                                                \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        verify##NAME##Attr(v, ir, *v0, *v1, *v2, *v3);                         \
    }                                                                          \
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
    detail::addAttr(ir, IRNAME, attrVals);                                     \
  }                                                                            \
                                                                               \
  void llvm::verify##NAME##Attr(KitVerifier &v, const IRELEM &ir) {            \
    if (const MDNode *attr =                                                   \
            detail::getRawAttr(IRNAME, detail::getRawAttrList(ir))) {          \
      if (!detail::verifyRawAttrValueCount(v, *attr, 5))                       \
        return;                                                                \
                                                                               \
      std::optional<ETY0> v0 = detail::getRawAttrValue<ETY0>(*attr, 0);        \
      std::optional<ETY1> v1 = detail::getRawAttrValue<ETY1>(*attr, 1);        \
      std::optional<ETY2> v2 = detail::getRawAttrValue<ETY2>(*attr, 2);        \
      std::optional<ETY3> v3 = detail::getRawAttrValue<ETY3>(*attr, 3);        \
      std::optional<ETY4> v4 = detail::getRawAttrValue<ETY4>(*attr, 4);        \
      if (!detail::verifyRawAttrValues(v, *attr, v0, v1, v2, v3, v4))          \
        return;                                                                \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        verify##NAME##Attr(v, ir, *v0, *v1, *v2, *v3, *v4);                    \
    }                                                                          \
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
    detail::addAttr(ir, IRNAME, attrVals);                                     \
  }                                                                            \
                                                                               \
  void llvm::verify##NAME##Attr(KitVerifier &v, const IRELEM &ir) {            \
                                                                               \
    if (const MDNode *attr =                                                   \
            detail::getRawAttr(IRNAME, detail::getRawAttrList(ir))) {          \
      if (!detail::verifyRawAttrValueCount(v, *attr, 6))                       \
        return;                                                                \
                                                                               \
      std::optional<ETY0> v0 = detail::getRawAttrValue<ETY0>(*attr, 0);        \
      std::optional<ETY1> v1 = detail::getRawAttrValue<ETY1>(*attr, 1);        \
      std::optional<ETY2> v2 = detail::getRawAttrValue<ETY2>(*attr, 2);        \
      std::optional<ETY3> v3 = detail::getRawAttrValue<ETY3>(*attr, 3);        \
      std::optional<ETY4> v4 = detail::getRawAttrValue<ETY4>(*attr, 4);        \
      std::optional<ETY5> v5 = detail::getRawAttrValue<ETY5>(*attr, 5);        \
      if (!detail::verifyRawAttrValues(v, *attr, v0, v1, v2, v3, v4, v5))      \
        return;                                                                \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        verify##NAME##Attr(v, ir, *v0, *v1, *v2, *v3, *v4, *v5);               \
    }                                                                          \
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
    detail::addAttr(ir, IRNAME, attrVals);                                     \
  }                                                                            \
                                                                               \
  void llvm::verify##NAME##Attr(KitVerifier &v, const IRELEM &ir) {            \
    if (const MDNode *attr =                                                   \
            detail::getRawAttr(IRNAME, detail::getRawAttrList(ir))) {          \
      if (!detail::verifyRawAttrValueCount(v, *attr, 7))                       \
        return;                                                                \
                                                                               \
      std::optional<ETY0> v0 = detail::getRawAttrValue<ETY0>(*attr, 0);        \
      std::optional<ETY1> v1 = detail::getRawAttrValue<ETY1>(*attr, 1);        \
      std::optional<ETY2> v2 = detail::getRawAttrValue<ETY2>(*attr, 2);        \
      std::optional<ETY3> v3 = detail::getRawAttrValue<ETY3>(*attr, 3);        \
      std::optional<ETY4> v4 = detail::getRawAttrValue<ETY4>(*attr, 4);        \
      std::optional<ETY5> v5 = detail::getRawAttrValue<ETY5>(*attr, 5);        \
      std::optional<ETY6> v6 = detail::getRawAttrValue<ETY6>(*attr, 6);        \
      if (!detail::verifyRawAttrValues(v, *attr, v0, v1, v2, v3, v4, v5, v6))  \
        return;                                                                \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        verify##NAME##Attr(v, ir, *v0, *v1, *v2, *v3, *v4, *v5, *v6);          \
    }                                                                          \
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
    detail::addAttr(ir, IRNAME, attrVals);                                     \
  }                                                                            \
                                                                               \
  void llvm::verify##NAME##Attr(KitVerifier &v, const IRELEM &ir) {            \
    if (const MDNode *attr =                                                   \
            detail::getRawAttr(IRNAME, detail::getRawAttrList(ir))) {          \
      if (!detail::verifyRawAttrValueCount(v, *attr, 8))                       \
        return;                                                                \
                                                                               \
      std::optional<ETY0> v0 = detail::getRawAttrValue<ETY0>(*attr, 0);        \
      std::optional<ETY1> v1 = detail::getRawAttrValue<ETY1>(*attr, 1);        \
      std::optional<ETY2> v2 = detail::getRawAttrValue<ETY2>(*attr, 2);        \
      std::optional<ETY3> v3 = detail::getRawAttrValue<ETY3>(*attr, 3);        \
      std::optional<ETY4> v4 = detail::getRawAttrValue<ETY4>(*attr, 4);        \
      std::optional<ETY5> v5 = detail::getRawAttrValue<ETY5>(*attr, 5);        \
      std::optional<ETY6> v6 = detail::getRawAttrValue<ETY6>(*attr, 6);        \
      std::optional<ETY7> v7 = detail::getRawAttrValue<ETY7>(*attr, 7);        \
      if (!detail::verifyRawAttrValues(v, *attr, v0, v1, v2, v3, v4, v5, v6,   \
                                       v7))                                    \
        return;                                                                \
                                                                               \
      if constexpr (CUSTOMVERIFY)                                              \
        llvm::verify##NAME##Attr(v, ir, *v0, *v1, *v2, *v3, *v4, *v5, *v6,     \
                                 *v7);                                         \
    }                                                                          \
  }

#define DEFN_ATTR_N(IRELEM, NAME, IRNAME, ETY, ENAME, EN, NELEMS)              \
  std::optional<ETY> llvm::get##ENAME##From##NAME##Attr(const IRELEM &ir) {    \
    if (const MDNode *attr =                                                   \
            detail::getRawAttr(IRNAME, detail::getRawAttrList(ir)))            \
      if (attr->getNumOperands() == NELEMS + 1)                                \
        return fromMetadata<ETY>(attr->getOperand(EN + 1));                    \
    return std::nullopt;                                                       \
  }

#endif // KITSUNE_LIB_CORE_ATTRS_IMPL_H

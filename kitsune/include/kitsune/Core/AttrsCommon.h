//==- AttrsCommon.h - Utilities common to Kitsune-specific attrs -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities shared by Kitsune-specific attributes.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_ATTRS_COMMON_H
#define KITSUNE_CORE_ATTRS_COMMON_H

#include "kitsune/Core/MetadataUtils.h"
#include "kitsune/Support/ToString.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Metadata.h"

namespace llvm {

class Loop;
class LoopInfo;

/// \addtogroup kitsune
/// @{

class KitVerifier;

/// Iterator over a raw attribute list.
class AttrIterator {
public:
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = MDNode &;
  using pointer = MDNode *;
  using reference = MDNode &;

public:
  AttrIterator() : attrList(nullptr), curr(0) {}
  AttrIterator(const MDNode *attrList) : attrList(attrList), curr(1) {}
  AttrIterator(const MDNode *attrList, unsigned last)
      : attrList(attrList), curr(last) {}

  reference operator*() const {
    return *cast<MDNode>(attrList->getOperand(curr));
  }

  pointer operator->() const {
    return cast<MDNode>(attrList->getOperand(curr));
  }

  AttrIterator &operator++() {
    curr++;
    return *this;
  }

  AttrIterator operator++(int) {
    AttrIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  friend bool operator==(const AttrIterator &l, const AttrIterator &r) {
    return l.attrList == r.attrList && l.curr == r.curr;
  }

  friend bool operator!=(const AttrIterator &l, const AttrIterator &r) {
    return !(l == r);
  }

private:
  const MDNode *attrList = nullptr;
  unsigned curr = 0;
};

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

// /// Get the value of the attribute \p attrName in the attribute list
// /// \p attrList. This expects the attribute to be single-valued where the value
// /// of the attribute is an LLVM loop.
// std::optional<Loop *>
// getAttrValue(StringRef attrName, const MDNode *attrList,
//              const SmallVectorImpl<const LoopInfo *> &lis);

// /// Parse the \p i 'th value from the metadata node for the attribute \p attr
// /// in the attribute list \p attrList. \p attrList may be nullptr, in which case
// /// this will return std::nullopt. i must be in [0, \p vals) where \p vals is
// /// the expected number of values permitted for the attribute. If \p vals is 0,
// /// this will always return std::nullopt.
// template <typename T>
// std::optional<T> getAttrValue(StringRef attrName, const MDNode *attrList,
//                               unsigned valNo, unsigned vals);

/// @}

} // namespace llvm

#define DECL_ATTR_COMMON(IRELEM, NAME, IRNAME, CUSTOMVERIFY, TYPE)             \
  bool has##NAME##Attr(const IRELEM &ir);                                      \
  void remove##NAME##Attr(IRELEM &ir);                                         \
  bool verify##NAME##Attr(KitVerifier &v, const IRELEM &ir);

#define DECL_ATTR_LOOP(IRELEM, NAME, IRNAME, CUSTOMVERIFY)                     \
  std::optional<Loop *> get##NAME##Attr(                                       \
      const IRELEM &ir, const SmallVectorImpl<const LoopInfo *> &lis);         \
  void add##NAME##Attr(IRELEM &ir, const Loop &loop);                          \
  bool verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, const MDNode &md);

#define DECL_ATTR_N(IRELEM, NAME, IRNAME, ETY, ENAME, EN, NELEMS)              \
  std::optional<ETY> get##ENAME##From##NAME##Attr(const IRELEM &);

#define DECL_ATTR_0(IRELEM, NAME, IRNAME, CUSTOMVERIFY)                        \
  void add##NAME##Attr(IRELEM &ir);                                            \
  bool verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, const bool &t);

#define DECL_ATTR_1(IRELEM, NAME, IRNAME, CUSTOMVERIFY, TYPE)                  \
  void add##NAME##Attr(IRELEM &ir, const TYPE &val);                           \
  std::optional<TYPE> get##NAME##Attr(const IRELEM &f);                        \
  bool verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, const TYPE &val);

#define DECL_ATTR_2(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1)                                         \
  void add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1);            \
  bool verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, const ETY0 &v0,    \
                          const ETY1 &v1);

#define DECL_ATTR_3(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2)                      \
  void add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,             \
                       const ETY2 &e2);                                        \
  bool verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, const ETY0 &v0,    \
                          const ETY1 &v1, const ETY2 &v2);

#define DECL_ATTR_4(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3)   \
  void add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,             \
                       const ETY2 &e2, const ETY3 &e3);                        \
  bool verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, const ETY0 &v0,    \
                          const ETY1 &v1, const ETY2 &v2, const ETY3 &v3);

#define DECL_ATTR_5(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4)                                         \
  void add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,             \
                       const ETY2 &e2, const ETY3 &e3, const ETY4 &e4);        \
  bool verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, const ETY0 &v0,    \
                          const ETY1 &v1, const ETY2 &v2, const ETY3 &v3,      \
                          const ETY4 &v4);

#define DECL_ATTR_6(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4, ETY5, ENAME5, EN5)                      \
  void add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,             \
                       const ETY2 &e2, const ETY3 &e3, const ETY4 &e4,         \
                       const ETY5 &e5);                                        \
  bool verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, const ETY0 &v0,    \
                          const ETY1 &v1, const ETY2 &v2, const ETY3 &v3,      \
                          const ETY4 &v4, const ETY5 &v5);

#define DECL_ATTR_7(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4, ETY5, ENAME5, EN5, ETY6, ENAME6, EN6)   \
  void add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,             \
                       const ETY2 &e2, const ETY3 &e3, const ETY4 &e4,         \
                       const ETY5 &e5, const ETY6 &e6);                        \
  bool verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, const ETY0 &v0,    \
                          const ETY1 &v1, const ETY2 &v2, const ETY3 &v3,      \
                          const ETY4 &v4, const ETY5 &v5, const ETY6 &v6);

#define DECL_ATTR_8(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4, ETY5, ENAME5, EN5, ETY6, ENAME6, EN6,   \
                    ETY7, ENAME7, EN7)                                         \
  void add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,             \
                       const ETY2 &e2, const ETY3 &e3, const ETY4 &e4,         \
                       const ETY5 &e5, const ETY6 &e6, const ETY7 &e7);        \
  bool verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, const ETY0 &v0,    \
                          const ETY1 &v1, const ETY2 &v2, const ETY3 &v3,      \
                          const ETY4 &v4, const ETY5 &v5, const ETY6 &v6,      \
                          const ETY7 &v7);

#endif // KITSUNE_CORE_ATTRS_COMMON_H

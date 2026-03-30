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
#include "kitsune/Core/VerifierInternal.h"
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

namespace detail {

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

/// Verify that the raw attribute \p attr has the expected number of values,
/// \p attrVals. If so, return true. Otherwise, if an optional output stream,
/// \p os, has been provided, write an error message to it.
bool verifyRawAttrValueCount(KitVerifier &v, const MDNode &attr,
                             unsigned attrVals);

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
std::optional<T> getRawAttrValue(const MDNode &attr, size_t i);

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

/// If the attribute list \p attrList contains an attribute \p attrName, return
/// the MDNode for that attribute. Otherwise, return nullptr. If found, the
/// MDNode that is returned will have at least one operand. This will be an
/// MDString whose value is the name of the attribute. If any other operands
/// are present, they will be the values accepted by the attribute. If
/// \p attrList is nullptr, this will also return nullptr.
MDNode *getRawAttr(StringRef attrName, const MDNode *attrList);

/// Get the value of the attribute \p attrName in the attribute list
/// \p attrList. This expects the attribute to be single-valued where the value
/// of the attribute is an LLVM loop.
std::optional<Loop *>
getAttrValue(StringRef attrName, const MDNode *attrList,
             const SmallVectorImpl<const LoopInfo *> &lis);

/// Parse the \p i 'th value from the metadata node for the attribute \p attr
/// in the attribute list \p attrList. \p attrList may be nullptr, in which case
/// this will return std::nullopt. i must be in [0, \p vals) where \p vals is
/// the expected number of values permitted for the attribute. If \p vals is 0,
/// this will always return std::nullopt.
template <typename T>
std::optional<T> getAttrValue(StringRef attrName, const MDNode *attrList,
                              unsigned valNo, unsigned vals);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_ATTRS_COMMON_H

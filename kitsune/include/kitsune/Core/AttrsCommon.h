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

class LLVMContext;
class Loop;
class LoopInfo;
class MDNode;
class Metadata;

#define ADD_0(IRNAME, OBJ)                                                     \
  do {                                                                         \
    ::addAttr(OBJ, IRNAME, {});                                                \
  } while (0)

#define ADD_1(IRNAME, OBJ, E0)                                                 \
  do {                                                                         \
    LLVMContext &ctx = getContext(OBJ);                                        \
    Metadata *attrVals[] = {toMetadata(E0, ctx)};                              \
    ::addAttr(OBJ, IRNAME, attrVals);                                          \
  } while (0)

#define ADD_2(IRNAME, OBJ, E0, E1)                                             \
  do {                                                                         \
    LLVMContext &ctx = getContext(OBJ);                                        \
    Metadata *attrVals[] = {toMetadata(E0, ctx), toMetadata(E1, ctx)};         \
    ::addAttr(OBJ, IRNAME, attrVals);                                          \
  } while (0)

#define ADD_3(IRNAME, OBJ, E0, E1, E2)                                         \
  do {                                                                         \
    LLVMContext &ctx = getContext(OBJ);                                        \
    Metadata *attrVals[] = {toMetadata(E0, ctx), toMetadata(E1, ctx),          \
                            toMetadata(E2, ctx)};                              \
    ::addAttr(OBJ, IRNAME, attrVals);                                          \
  } while (0)

#define ADD_4(IRNAME, OBJ, E0, E1, E2, E3)                                     \
  do {                                                                         \
    LLVMContext &ctx = getContext(OBJ);                                        \
    Metadata *attrVals[] = {toMetadata(E0, ctx), toMetadata(E1, ctx),          \
                            toMetadata(E2, ctx), toMetadata(E3, ctx)};         \
    ::addAttr(OBJ, IRNAME, attrVals);                                          \
  } while (0)

#define ADD_5(IRNAME, OBJ, E0, E1, E2, E3, E4)                                 \
  do {                                                                         \
    LLVMContext &ctx = getContext(OBJ);                                        \
    Metadata *attrVals[] = {toMetadata(E0, ctx), toMetadata(E1, ctx),          \
                            toMetadata(E2, ctx), toMetadata(E3, ctx),          \
                            toMetadata(E4, ctx)};                              \
    ::addAttr(OBJ, IRNAME, attrVals);                                          \
  } while (0)

#define ADD_6(IRNAME, OBJ, E0, E1, E2, E3, E4, E5)                             \
  do {                                                                         \
    LLVMContext &ctx = getContext(OBJ);                                        \
    Metadata *attrVals[] = {toMetadata(E0, ctx), toMetadata(E1, ctx),          \
                            toMetadata(E2, ctx), toMetadata(E3, ctx),          \
                            toMetadata(E4, ctx), toMetadata(E5, ctx)};         \
    ::addAttr(OBJ, IRNAME, attrVals);                                          \
  } while (0)

#define ADD_7(IRNAME, OBJ, E0, E1, E2, E3, E4, E5, E6)                         \
  do {                                                                         \
    LLVMContext &ctx = getContext(OBJ);                                        \
    Metadata *attrVals[] = {toMetadata(E0, ctx), toMetadata(E1, ctx),          \
                            toMetadata(E2, ctx), toMetadata(E3, ctx),          \
                            toMetadata(E4, ctx), toMetadata(E5, ctx),          \
                            toMetadata(E6, ctx)};                              \
    ::addAttr(OBJ, IRNAME, attrVals);                                          \
  } while (0)

#define ADD_8(IRNAME, OBJ, E0, E1, E2, E3, E4, E5, E6, E7)                     \
  do {                                                                         \
    LLVMContext &ctx = getContext(OBJ);                                        \
    Metadata *attrVals[] = {toMetadata(E0, ctx), toMetadata(E1, ctx),          \
                            toMetadata(E2, ctx), toMetadata(E3, ctx),          \
                            toMetadata(E4, ctx), toMetadata(E5, ctx),          \
                            toMetadata(E6, ctx), toMetadata(E7, ctx)};         \
    ::addAttr(OBJ, IRNAME, attrVals);                                          \
  } while (0)

#define VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETYPE, ENAME)                       \
  do {                                                                         \
    if (!get##ENAME##From##NAME##Attr(OBJ)) {                                  \
      if (OS)                                                                  \
        (*OS) << "Missing value of type '" << toString<ETYPE>()                \
              << "' for element '" << #ENAME << "' in attribute '" << IRNAME   \
              << "'\n";                                                        \
      return false;                                                            \
    }                                                                          \
  } while (0)

#define VERIFY_1(OS, OBJ, NAME, IRNAME, TYPE)                                  \
  do {                                                                         \
    if (has##NAME##Attr(OBJ)) {                                                \
      if (!get##NAME##Attr(OBJ).has_value()) {                                 \
        if (OS)                                                                \
          (*OS) << "Missing value of type '" << toString<TYPE>()               \
                << "' in attribute '" << IRNAME << "'\n";                      \
        return false;                                                          \
      }                                                                        \
    }                                                                          \
    return true;                                                               \
  } while (0)

#define VERIFY_2(OS, OBJ, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1)            \
  do {                                                                         \
    if (has##NAME##Attr(OBJ)) {                                                \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY0, ENAME0);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY1, ENAME1);                        \
    }                                                                          \
    return true;                                                               \
  } while (0)

#define VERIFY_3(OS, OBJ, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2,      \
                 ENAME2)                                                       \
  do {                                                                         \
    if (has##NAME##Attr(OBJ)) {                                                \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY0, ENAME0);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY1, ENAME1);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY2, ENAME2);                        \
    }                                                                          \
    return true;                                                               \
  } while (0)

#define VERIFY_4(OS, OBJ, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2,      \
                 ENAME2, ETY3, ENAME3)                                         \
  do {                                                                         \
    if (has##NAME##Attr(OBJ)) {                                                \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY0, ENAME0);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY1, ENAME1);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY2, ENAME2);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY3, ENAME3);                        \
    }                                                                          \
    return true;                                                               \
  } while (0)

#define VERIFY_5(OS, OBJ, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2,      \
                 ENAME2, ETY3, ENAME3, ETY4, ENAME4)                           \
  do {                                                                         \
    if (has##NAME##Attr(OBJ)) {                                                \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY0, ENAME0);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY1, ENAME1);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY2, ENAME2);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY3, ENAME3);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY4, ENAME4);                        \
    }                                                                          \
    return true;                                                               \
  } while (0)

#define VERIFY_6(OS, OBJ, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2,      \
                 ENAME2, ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5)             \
  do {                                                                         \
    if (has##NAME##Attr(OBJ)) {                                                \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY0, ENAME0);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY1, ENAME1);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY2, ENAME2);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY3, ENAME3);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY4, ENAME4);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY5, ENAME5);                        \
    }                                                                          \
    return true;                                                               \
  } while (0)

#define VERIFY_7(OS, OBJ, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2,      \
                 ENAME2, ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5, ETY6,       \
                 ENAME6)                                                       \
  do {                                                                         \
    if (has##NAME##Attr(OBJ)) {                                                \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY0, ENAME0);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY1, ENAME1);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY2, ENAME2);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY3, ENAME3);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY4, ENAME4);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY5, ENAME5);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY6, ENAME6);                        \
    }                                                                          \
    return true;                                                               \
  } while (0)

#define VERIFY_8(OS, OBJ, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2,      \
                 ENAME2, ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5, ETY6,       \
                 ENAME6, ETY7, ENAME7)                                         \
  do {                                                                         \
    if (has##NAME##Attr(OBJ)) {                                                \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY0, ENAME0);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY1, ENAME1);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY2, ENAME2);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY3, ENAME3);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY4, ENAME4);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY5, ENAME5);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY6, ENAME6);                        \
      VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY7, ENAME7);                        \
    }                                                                          \
    return true;                                                               \
  } while (0)

/// Create a new empty attribute list. This will be of the form
///
/// \code{llvm}
///     !0 = distinct !{!0}
/// \endcode
MDNode *getNewAttrList(LLVMContext &ctx);

/// Create a new attribute list containing the attribute with name \p attrName
/// and values \p attrVals. \p attrList is the existing attribute list. It may
/// be null in which case the returned list will contain a single attribute. If
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
///     getNewAttrListWithAttr("new-attr", {...}, nullptr, ctx);
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
///     getNewAttrListWithAttr("new-attr", {...}, <attrList>, ctx);
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
///     getNewAttrListWithAttr("new-attr", {!"new"}, <attrList>, ctx);
/// \endcode
///
/// \code{llvm}
///     !0 = distinct !{!0, !1, !2}
///     !1 = !{!"attr-flag"}
///     !2 = !{!"new-attr", !"new"}
/// \endcode
///
MDNode *getNewAttrListWith(StringRef attrName,
                           const ArrayRef<Metadata *> attrVals,
                           const MDNode *attrList, LLVMContext &ctx);

/// Remove the attribute named \p attrName from \p attrList. If the attribute
/// exists in the list, a new MDNode will be created and returned. Otherwise,
/// \p attrList will be returned. If removing the result would result in an
/// empty list, return nullptr. If \p attrList is nullptr, returns nullptr.
MDNode *getNewAttrListWithout(StringRef attrName, MDNode *attrList);

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
                              unsigned valNo, unsigned vals) {
  if (attrList && vals && valNo < vals)
    if (MDNode *md = getRawAttr(attrName, attrList))
      // The first operand of the metadata node will be the name of the
      // attribute.
      if (md->getNumOperands() == vals + 1)
        return fromMetadata<T>(md->getOperand(valNo + 1));
  return std::nullopt;
}

/// Verify an attribute \p attrName that is not expected to have any values.
/// Return false if the \p attrName is present in \p attrList and has one or
/// more values. Return true in all other cases, including when \p attrList is
/// nullptr, and \p attrName is not present in \p attrList. If false is due to
/// be returned, and the optional output stream \p os is not nullptr, print an
/// error message to it.
bool verifyAttr0(StringRef attrName, const MDNode *attrList,
                 raw_ostream *os = nullptr);

/// Verify an attribute \p attrName that is expected to have a single value.
/// This value is an MDNode that corresponds to the ID of a loop. Return false
/// if \p attrName is present in \p attrList and does not have exactly one
/// value. Without a LoopInfo object, it is impossible to truly verify that the
/// value is the ID of a loop. Instead, some rudimentary checks are performed -
/// in particular that the MDNode is distinct and the first operand is a
/// self-reference. If any of these is not the case, return false. Return true
/// in all other cases, including when \p attrList is nullptr, and \p attrName
/// is not present in \p attrList. If false is due to be returned, and the
/// optional output stream \p os is not nullptr, print an error message to it.
bool verifyAttrLoop(StringRef attrName, const MDNode *attrList,
                    raw_ostream *os = nullptr);

} // namespace llvm

#endif // KITSUNE_CORE_ATTRS_COMMON_H

//=- GVAttrsImpl.h -  Kitsune-specific global variable attributes -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private header with declarations for the core implementation of
// Kitsune-specific global variable attributes. These intentionally "leak" so
// they can be shared with the unit tests.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_CORE_GV_ATTRS_IMPL_H
#define KITSUNE_LIB_CORE_GV_ATTRS_IMPL_H

#include "AttrsImplDeclMacros.h"
#include "AttrsIterator.h"
#include "kitsune/Core/TTID.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"

namespace llvm {

class GlobalVariable;
class MDNode;
class Metadata;

class KitVerifier;

namespace detail {

/// Get the MDNode, if it exists, for an attribute named \p attrName for the
/// global variable \p g. Otherwise, return nullptr.
MDNode *getRawAttr(const GlobalVariable &g, StringRef attrName);

/// Get the raw list of Kitsune-specific attributes from a global variable.
MDNode *getRawAttrList(const GlobalVariable &g);

/// Update the raw list of Kitsune-specific attributes in a global variable with
/// the new list \p attrList. If \p attrList is nullptr, the attribute list will
/// be removed from the global variable.
void setAttrList(GlobalVariable &g, MDNode *attrList);

/// Add an attribute named \p attrName with values \p attrVals to a global
/// variable.
void addAttr(GlobalVariable &g, StringRef attrName,
             ArrayRef<Metadata *> attrVals);

/// Remove the attribute named \p attrName from a global variable. If the
/// attribute is not already on the global variable, this has no effect.
void removeAttr(GlobalVariable &g, StringRef attrName);

/// Get a pair of iterators over the raw Kitsune-specific global variable
/// attributes.
iterator_range<AttrIterator> attrs(const GlobalVariable &g);

/// Verify the attribute named \p attrName that is known to be on the global
/// variable \p g. This will only set the internal state in the verifier \p v.
void verifyAttr(KitVerifier &v, const GlobalVariable &g, StringRef attrName);

} // namespace detail

#define GV_ATTR(...) DECL_ATTR_IMPL_COMMON(GlobalVariable, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

#define GV_ATTR_L(...) DECL_ATTR_IMPL_L(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_S(...) DECL_ATTR_IMPL_S(GlobalVariable, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

#define GV_ATTR_0(...) DECL_ATTR_IMPL_0(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_1(...) DECL_ATTR_IMPL_1(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_2(...) DECL_ATTR_IMPL_2(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_3(...) DECL_ATTR_IMPL_3(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_4(...) DECL_ATTR_IMPL_4(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_5(...) DECL_ATTR_IMPL_5(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_6(...) DECL_ATTR_IMPL_6(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_7(...) DECL_ATTR_IMPL_7(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_8(...) DECL_ATTR_IMPL_8(GlobalVariable, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

} // namespace llvm

#endif // KITSUNE_LIB_CORE_GV_ATTRS_IMPL_H

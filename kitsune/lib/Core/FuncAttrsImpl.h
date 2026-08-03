//===- FuncAttrsImpl.h -  Kitsune-specific function attributes --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private header with declarations for the core implementation of
// Kitsune-specific function attributes. These intentionally "leak" so they can
// be shared with the unit tests.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_CORE_FUNC_ATTRS_IMPL_H
#define KITSUNE_LIB_CORE_FUNC_ATTRS_IMPL_H

#include "AttrsImplDeclMacros.h"
#include "AttrsIterator.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"

namespace llvm {

class Function;
class MDNode;
class Metadata;

class KitVerifier;

namespace detail {

/// Get the raw list of Kitsune-specific attributes from a function.
MDNode *getRawAttrList(const Function &f);

/// Update the raw list of Kitsune-specific attributes in a function with the
/// new list \p attrList. If \p attrList is nullptr, the attribute list will be
/// removed from the function.
void setAttrList(Function &f, MDNode *attrList);

/// Add an attribute named \p attrName with values \p attrVals to a function.
void addAttr(Function &f, StringRef attrName, ArrayRef<Metadata *> attrVals);

/// Remove the attribute named \p attrName from a function. If the attribute is
/// not already on the function, this has no effect.
void removeAttr(Function &f, StringRef attrName);

/// Get a pair of iterators over the raw Kitsune-specific function attributes.
iterator_range<AttrIterator> attrs(const Function &f);

/// Verify the attribute named \p attrName that is known to be on the function
/// \p f. This will only set the internal state in the verifier \p v.
void verifyAttr(KitVerifier &v, const Function &f, StringRef attrName);

} // namespace detail

#define FUNC_ATTR(...) DECL_ATTR_IMPL_COMMON(Function, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

#define FUNC_ATTR_L(...) DECL_ATTR_IMPL_L(Function, __VA_ARGS__)
#define FUNC_ATTR_S(...) DECL_ATTR_IMPL_S(Function, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

#define FUNC_ATTR_0(...) DECL_ATTR_IMPL_0(Function, __VA_ARGS__)
#define FUNC_ATTR_1(...) DECL_ATTR_IMPL_1(Function, __VA_ARGS__)
#define FUNC_ATTR_2(...) DECL_ATTR_IMPL_2(Function, __VA_ARGS__)
#define FUNC_ATTR_3(...) DECL_ATTR_IMPL_3(Function, __VA_ARGS__)
#define FUNC_ATTR_4(...) DECL_ATTR_IMPL_4(Function, __VA_ARGS__)
#define FUNC_ATTR_5(...) DECL_ATTR_IMPL_5(Function, __VA_ARGS__)
#define FUNC_ATTR_6(...) DECL_ATTR_IMPL_6(Function, __VA_ARGS__)
#define FUNC_ATTR_7(...) DECL_ATTR_IMPL_7(Function, __VA_ARGS__)
#define FUNC_ATTR_8(...) DECL_ATTR_IMPL_8(Function, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

} // namespace llvm

#endif // KITSUNE_LIB_CORE_FUNC_ATTRS_IMPL_H

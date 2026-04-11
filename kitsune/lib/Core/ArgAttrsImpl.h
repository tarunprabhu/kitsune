//===- ArgAttrsImpl.h -  Kitsune-specific argument attributes =--*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private header with declarations for the core implementation of
// Kitsune-specific function argument attributes. These intentionally "leak" so
// they can be shared with the unit tests.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_CORE_ARG_ATTRS_IMPL_H
#define KITSUNE_LIB_CORE_ARG_ATTRS_IMPL_H

#include "AttrsImplDeclMacros.h"
#include "AttrsIterator.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"

namespace llvm {

class Argument;
class MDNode;
class Metadata;

class KitVerifier;

namespace detail {

/// Get the raw list of Kitsune-specific attributes for the given argument. The
/// argument must have a parent.
MDNode *getRawAttrList(const Argument &a);

/// Update the raw list of Kitsune-specific attributes for function arguments
/// with the new list \p attrList. If \p attrList is nullptr, the attribute list
/// will be removed from the argument. The argument must have a parent.
void setAttrList(Argument &a, MDNode *attrList);

/// Add an attribute named \p attrName with values \p attrVals to an argument.
/// The argument must have a parent.
void addAttr(Argument &a, StringRef attrName, ArrayRef<Metadata *> attrVals);

/// Remove the attribute named \p attrName from an argument. If the attribute is
/// not already on the argument, this has no effect.
void removeAttr(Argument &a, StringRef attrName);

/// Get a pair of iterators over the raw Kitsune-specific argument attributes.
iterator_range<AttrIterator> attrs(const Argument &a);

/// Verify the attribute named \p attrName that is known to be on the argument
/// \p a. This will only set the internal state in the verifier \p v.
void verifyAttr(KitVerifier &v, const Argument &a, StringRef attrName);

} // namespace detail

#define ARG_ATTR(...) DECL_ATTR_IMPL_COMMON(Argument, __VA_ARGS__)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"

#define ARG_ATTR_0(...) DECL_ATTR_IMPL_0(Argument, __VA_ARGS__)
#define ARG_ATTR_1(...) DECL_ATTR_IMPL_1(Argument, __VA_ARGS__)
#define ARG_ATTR_2(...) DECL_ATTR_IMPL_2(Argument, __VA_ARGS__)
#define ARG_ATTR_3(...) DECL_ATTR_IMPL_3(Argument, __VA_ARGS__)
#define ARG_ATTR_4(...) DECL_ATTR_IMPL_4(Argument, __VA_ARGS__)
#define ARG_ATTR_5(...) DECL_ATTR_IMPL_5(Argument, __VA_ARGS__)
#define ARG_ATTR_6(...) DECL_ATTR_IMPL_6(Argument, __VA_ARGS__)
#define ARG_ATTR_7(...) DECL_ATTR_IMPL_7(Argument, __VA_ARGS__)
#define ARG_ATTR_8(...) DECL_ATTR_IMPL_8(Argument, __VA_ARGS__)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"

} // namespace llvm

#endif // KITSUNE_LIB_CORE_ARG_ATTRS_IMPL_H

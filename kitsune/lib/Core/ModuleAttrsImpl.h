//===- ModuleAttrsImpl.h -  Kitsune-specific module attributes --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private header with declarations for the core implementation of
// Kitsune-specific module attributes. These intentionally "leak" so they can be
// shared with the unit tests.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_CORE_MODULE_ATTRS_IMPL_H
#define KITSUNE_LIB_CORE_MODULE_ATTRS_IMPL_H

#include "AttrsImplDeclMacros.h"
#include "AttrsIterator.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"

namespace llvm {

class Module;
class MDNode;
class Metadata;

class KitVerifier;

namespace detail {

/// Get the MDNode, if it exists, for an attribute named \p attrName for the
/// module \p m. Otherwise, return nullptr.
MDNode *getRawAttr(const Module &m, StringRef attrName);

/// Get the raw list of Kitsune-specific attributes from a module.
MDNode *getRawAttrList(const Module &m);

/// Update the raw list of Kitsune-specific attributes in a module with the new
/// list \p attrList. If \p attrList is nullptr, the attribute list will be
/// removed from the module.
void setAttrList(Module &m, MDNode *attrList);

/// Add an attribute named \p attrName with values \p attrVals to a module.
void addAttr(Module &m, StringRef attrName, ArrayRef<Metadata *> attrVals);

/// Remove the attribute named \p attrName from a module. If the attribute is
/// not already on the module, this has no effect.
void removeAttr(Module &m, StringRef attrName);

/// Get a pair of iterators over the raw Kitsune-specific module attributes.
iterator_range<AttrIterator> attrs(const Module &m);

/// Verify the attribute named \p attrName that is known to be on the module
/// \p m. This will only set the internal state in the verifier \p v.
void verifyAttr(KitVerifier &v, const Module &m, StringRef attrName);

} // namespace detail

#define MODULE_ATTR(...) DECL_ATTR_IMPL_COMMON(Module, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

#define MODULE_ATTR_L(...) DECL_ATTR_IMPL_L(Module, __VA_ARGS__)
#define MODULE_ATTR_S(...) DECL_ATTR_IMPL_S(Module, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

#define MODULE_ATTR_0(...) DECL_ATTR_IMPL_0(Module, __VA_ARGS__)
#define MODULE_ATTR_1(...) DECL_ATTR_IMPL_1(Module, __VA_ARGS__)
#define MODULE_ATTR_2(...) DECL_ATTR_IMPL_2(Module, __VA_ARGS__)
#define MODULE_ATTR_3(...) DECL_ATTR_IMPL_3(Module, __VA_ARGS__)
#define MODULE_ATTR_4(...) DECL_ATTR_IMPL_4(Module, __VA_ARGS__)
#define MODULE_ATTR_5(...) DECL_ATTR_IMPL_5(Module, __VA_ARGS__)
#define MODULE_ATTR_6(...) DECL_ATTR_IMPL_6(Module, __VA_ARGS__)
#define MODULE_ATTR_7(...) DECL_ATTR_IMPL_7(Module, __VA_ARGS__)
#define MODULE_ATTR_8(...) DECL_ATTR_IMPL_8(Module, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

} // namespace llvm

#endif // KITSUNE_LIB_CORE_MODULE_ATTRS_IMPL_H

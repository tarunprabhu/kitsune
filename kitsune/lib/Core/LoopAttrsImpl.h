//===- LoopAttrsImpl.h -  Kitsune-specific loop attributes ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private header with declarations for the core implementation of
// Kitsune-specific loop attributes. These intentionally "leak" so they can be
// shared with the unit tests.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_CORE_LOOP_ATTRS_IMPL_H
#define KITSUNE_LIB_CORE_LOOP_ATTRS_IMPL_H

#include "AttrsImplDeclMacros.h"
#include "AttrsIterator.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"

namespace llvm {

class Loop;
class MDNode;
class Metadata;

class KitVerifier;

namespace detail {

/// Get the raw list of attributes from a loop. The list will contain both
/// Kitsune-specific and core LLVM "attributes".
MDNode *getRawAttrList(const Loop &loop);

/// Update the list of attributes in a loop with the new list \p attrList. If
/// \p attrList is nullptr, the attribute list will be removed from the loop.
/// A null value of \p attrList will result in the loop ID metadata being
/// removed. This will likely break other passes since the loop analysis pass
/// may not work as expected after doing so.
void setAttrList(Loop &loop, MDNode *attrList);

/// Add an attribute named \p attrName with values \p attrVals to a loop.
void addAttr(Loop &loop, StringRef attrName, ArrayRef<Metadata *> attrVals);

/// Remove the attribute named \p attrName from a loop. If the attribute is not
/// already on the loop, this has no effect.
void removeAttr(Loop &loop, StringRef attrName);

/// Get a pair of iterators over over the raw loop attributes. This will
/// include both Kitsune-specific and core LLVM attributes.
iterator_range<AttrIterator> attrs(const Loop &loop);

/// Verify the attribute named \p attrName that is known to be on the loop
/// \p loop. This will only set the internal state in the verifier \p v.
void verifyAttr(KitVerifier &v, const Loop &loop, StringRef attrName);

} // namespace detail

#define LOOP_ATTR(...) DECL_ATTR_IMPL_COMMON(Loop, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

#define LOOP_ATTR_L(...) DECL_ATTR_IMPL_L(Loop, __VA_ARGS__)
#define LOOP_ATTR_S(...) DECL_ATTR_IMPL_S(Loop, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

#define LOOP_ATTR_0(...) DECL_ATTR_IMPL_0(Loop, __VA_ARGS__)
#define LOOP_ATTR_1(...) DECL_ATTR_IMPL_1(Loop, __VA_ARGS__)
#define LOOP_ATTR_2(...) DECL_ATTR_IMPL_2(Loop, __VA_ARGS__)
#define LOOP_ATTR_3(...) DECL_ATTR_IMPL_3(Loop, __VA_ARGS__)
#define LOOP_ATTR_4(...) DECL_ATTR_IMPL_4(Loop, __VA_ARGS__)
#define LOOP_ATTR_5(...) DECL_ATTR_IMPL_5(Loop, __VA_ARGS__)
#define LOOP_ATTR_6(...) DECL_ATTR_IMPL_6(Loop, __VA_ARGS__)
#define LOOP_ATTR_7(...) DECL_ATTR_IMPL_7(Loop, __VA_ARGS__)
#define LOOP_ATTR_8(...) DECL_ATTR_IMPL_8(Loop, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

} // namespace llvm

#endif // KITSUNE_LIB_CORE_LOOP_ATTRS_IMPL_H

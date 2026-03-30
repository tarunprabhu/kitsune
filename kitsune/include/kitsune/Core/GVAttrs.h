//===- GVAttrs.h - Kitsune-specific attributes for global vars --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions and utilities to work with Kitsune-specific attributes for global
// variables.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_GV_ATTRS_H
#define KITSUNE_CORE_GV_ATTRS_H

#include "kitsune/Core/AttrsCommon.h"
#include "kitsune/Core/AttrsInternal.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class GlobalVariable;
class MDNode;
class raw_ostream;

/// \addtogroup kitsune
/// @{

/// Kitsune-specific attributes for global variables.
enum class GVAttrKind : uint32_t {
#define GET_GV_ATTR_ENUMS
#include "kitsune/Core/GVAttrs.inc"
};

/// Get the metadata node containing the list of Kitsune-specific attributes.
/// If no Kitsune-specific attributes have been added to the global variable,
/// this may return nullptr.
MDNode *getRawAttrList(const GlobalVariable &g);

/// Get the name of a global variable attribute as it would appear in LLVM
/// metadata. The result will start with "kit.gv.".
StringRef getAttrName(GVAttrKind attr);

/// Get the kind of a global variable attribute if the given string is how the
/// attribute would appear in LLVM metadata. Otherwise, return std::nullopt.
std::optional<GVAttrKind> getGVAttrKind(StringRef name);

/// Check if an attribute is present on a global variable.
bool hasAttr(const GlobalVariable &f, GVAttrKind attr);

/// Add an attribute to a global variable. Only attributes that do not take any
/// values can be added this way. Adding an attribute that takes values will
/// result in a catastrophic runtime error.
void addAttr(GlobalVariable &f, GVAttrKind attr);

/// Remove an attribute from a global variable. If the global variable does not
/// contain the attribute, this has no effect.
void removeAttr(GlobalVariable &f, GVAttrKind attr);

/// Verify an attribute \p attr on the global variable \p g. Returns true if any
/// of the following are true:
///
///   - \p attr is not present on \p g
///   - \p attr is present with the correct number of values, each of which is
///     of the correct type.
///
/// Otherwise, return false.
///
bool verifyAttr(KitVerifier &v, const GlobalVariable &g, GVAttrKind attr);

/// Get a range to iterate over the raw global variable attributes.
iterator_range<AttrIterator> attrs(const GlobalVariable &g);

/// @}

#define GV_ATTR(...) DECL_ATTR_COMMON(GlobalVariable, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

#define GV_ATTR_LOOP(...) DECL_ATTR_LOOP(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_0(...) DECL_ATTR_0(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_1(...) DECL_ATTR_1(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_2(...) DECL_ATTR_2(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_3(...) DECL_ATTR_3(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_4(...) DECL_ATTR_4(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_5(...) DECL_ATTR_5(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_6(...) DECL_ATTR_6(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_7(...) DECL_ATTR_7(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_8(...) DECL_ATTR_8(GlobalVariable, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

#define GV_ATTR_N(...) DECL_ATTR_N(GlobalVariable, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

} // namespace llvm

#endif // KITSUNE_CORE_GV_ATTRS_H

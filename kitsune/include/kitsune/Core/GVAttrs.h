//===- GVAttrs.h - Kitsune-specific attributes for global vars -*- C++ -*--===//
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

#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class GlobalVariable;

/// \addtogroup kitsune
/// \@{

/// Kitsune-specific attributes for functions.
enum class GVAttrKind : uint32_t {
#define GET_GV_ATTR_ENUMS
#include "kitsune/Core/GVAttrs.inc"
};

/// Get the name of the function attribute as it appears in the LLVM-IR
/// metadata. The result will start with "kit.func.".
StringRef getAttrName(GVAttrKind attr);

/// Get the kind of an function attribute if the given string corresponds to the
/// name of a function attribute as it might appear in LLVM-IR. If the string
/// does not correspond to a valid attribute name, return std::nullopt.
std::optional<GVAttrKind> getGVAttrKind(StringRef name);

/// Check if the given attribute is present on a function.
bool hasAttr(const GlobalVariable &f, GVAttrKind attr);

/// Add an attribute to the function. Only attributes that do not take any
/// values can be added using this function. Adding any other attribute will
/// result in a catastrophic runtime error.
void addAttr(GlobalVariable &f, GVAttrKind attr);

/// Remove the attribute from a function. If the function does not contain the
/// attribute, this has no effect.
void removeAttr(GlobalVariable &f, GVAttrKind attr);

/// @}

// Flag attributes (those that do not have a value) have a different set of
// accessors. Mask these before generating declarations for the other
// attributes.
#define GV_ATTR_FLAG(NAME, IRNAME)
#define GV_ATTR(NAME, TYPE, IRNAME)                                            \
  bool has##NAME##Attr(const GlobalVariable &g);                               \
  std::optional<TYPE> get##NAME##Attr(const GlobalVariable &g);                \
  void add##NAME##Attr(GlobalVariable &g, TYPE val);                           \
  void remove##NAME##Attr(GlobalVariable &g);
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

#define GV_ATTR_FLAG(NAME, IRNAME)                                             \
  bool has##NAME##Attr(const GlobalVariable &g);                               \
  void add##NAME##Attr(GlobalVariable &g);                                     \
  void remove##NAME##Attr(GlobalVariable &g);
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

} // namespace llvm

#endif // KITSUNE_CORE_GV_ATTRS_H

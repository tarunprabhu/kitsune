//===- FuncAttrs.h - Kitsune-specific attributes for functions -*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions and utilities to work with function attributes. These are not
// known to LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_FUNC_ATTRS_H
#define KITSUNE_CORE_FUNC_ATTRS_H

#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class Function;

/// \addtogroup kitsune
/// \@{

/// Kitsune-specific attributes for functions.
enum class FuncAttrKind : uint32_t {
#define GET_FUNC_ATTR_ENUMS
#include "kitsune/Core/FuncAttrs.inc"
};

/// Get the name of the function attribute as it appears in the LLVM-IR
/// metadata. The result will start with "kit.func.".
StringRef getAttrName(FuncAttrKind attr);

/// Get the kind of an function attribute if the given string corresponds to the
/// name of a function attribute as it might appear in LLVM-IR. If the string
/// does not correspond to a valid attribute name, return std::nullopt.
std::optional<FuncAttrKind> getFuncAttrKind(StringRef name);

/// Check if the given attribute is present on a function.
bool hasAttr(const Function &f, FuncAttrKind attr);

/// Add an attribute to the function. Only attributes that do not take any
/// values can be added using this function. Adding any other attribute will
/// result in a catastrophic runtime error.
void addAttr(Function &f, FuncAttrKind attr);

/// Remove the attribute from a function. If the function does not contain the
/// attribute, this has no effect.
void removeAttr(Function &f, FuncAttrKind attr);

/// @}

// Flag attributes (those that do not have a value) have a different set of
// accessors. Mask these before generating declarations for the other
// attributes.
#define FUNC_ATTR_FLAG(NAME, IRNAME)
#define FUNC_ATTR(NAME, TYPE, IRNAME)                                          \
  bool has##NAME##Attr(const Function &f);                                     \
  std::optional<TYPE> get##NAME##Attr(const Function &f);                      \
  void add##NAME##Attr(Function &f, TYPE val);                                 \
  void remove##NAME##Attr(Function &f);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

#define FUNC_ATTR_FLAG(NAME, IRNAME)                                           \
  bool has##NAME##Attr(const Function &f);                                     \
  void add##NAME##Attr(Function &f);                                           \
  void remove##NAME##Attr(Function &f);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

} // namespace llvm

#endif // KITSUNE_CORE_FUNC_ATTRS_H

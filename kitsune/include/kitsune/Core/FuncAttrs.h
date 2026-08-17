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

#include "kitsune/Core/AttrsDeclMacros.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class Function;
class raw_ostream;

/// \addtogroup kitsune
/// @{

/// Kitsune-specific attributes for functions.
enum class FuncAttrKind : uint32_t {
#define GET_FUNC_ATTR_ENUMS
#include "kitsune/Core/FuncAttrs.inc"
};

/// Stream the kind to LLVM's output stream.
raw_ostream &operator<<(raw_ostream &os, const FuncAttrKind &attr);

/// Get the name of a function attribute as it would appear in LLVM metadata.
StringRef getAttrName(FuncAttrKind attr);

/// Get the kind of an attribute if the given string is how the attribute would
/// appear in LLVM metadata. Otherwise, return std::nullopt.
std::optional<FuncAttrKind> getFuncAttrKind(StringRef name);

/// Check if an attribute is present on a function.
bool hasAttr(const Function &f, FuncAttrKind attr);

/// Add an attribute to a function. Only attributes that do not take any values
/// can be added this way. Adding an attribute that takes values will result in
/// a catastrophic runtime error.
void addAttr(Function &f, FuncAttrKind attr);

/// Remove an attribute from a function. If the function does not contain the
/// attribute, this has no effect.
void removeAttr(Function &f, FuncAttrKind attr);

/// @}

#define FUNC_ATTR(...) DECL_ATTR_COMMON(Function, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

#define FUNC_ATTR_L(...) DECL_ATTR_L(Function, __VA_ARGS__)
#define FUNC_ATTR_S(...) DECL_ATTR_S(Function, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

#define FUNC_ATTR_0(...) DECL_ATTR_0(Function, __VA_ARGS__)
#define FUNC_ATTR_1(...) DECL_ATTR_1(Function, __VA_ARGS__)
#define FUNC_ATTR_2(...) DECL_ATTR_2(Function, __VA_ARGS__)
#define FUNC_ATTR_3(...) DECL_ATTR_3(Function, __VA_ARGS__)
#define FUNC_ATTR_4(...) DECL_ATTR_4(Function, __VA_ARGS__)
#define FUNC_ATTR_5(...) DECL_ATTR_5(Function, __VA_ARGS__)
#define FUNC_ATTR_6(...) DECL_ATTR_6(Function, __VA_ARGS__)
#define FUNC_ATTR_7(...) DECL_ATTR_7(Function, __VA_ARGS__)
#define FUNC_ATTR_8(...) DECL_ATTR_8(Function, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

#define FUNC_ATTR_N(...) DECL_ATTR_N(Function, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

} // namespace llvm

#endif // KITSUNE_CORE_FUNC_ATTRS_H

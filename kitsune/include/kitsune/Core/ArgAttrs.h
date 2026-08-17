//===- ArgAttrs.h - Kitsune-specific attributes for arguments --*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to work with attributes for function arguments. These are not known
// to LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_ARG_ATTRS_H
#define KITSUNE_CORE_ARG_ATTRS_H

#include "kitsune/Core/AttrsDeclMacros.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class Argument;
class raw_ostream;

/// \addtogroup kitsune
/// @{

/// Kitsune-specific attributes for function argument.
enum class ArgAttrKind : uint32_t {
#define GET_ARG_ATTR_ENUMS
#include "kitsune/Core/ArgAttrs.inc"
};

/// Stream the kind to LLVM's output stream.
raw_ostream &operator<<(raw_ostream &os, const ArgAttrKind &attr);

/// Get the name of an argument attribute as it would appear in LLVM metadata.
StringRef getAttrName(ArgAttrKind attr);

/// Get the kind of an attribute if the given string is how the attribute would
/// appear in LLVM metadata. Otherwise, return std::nullopt.
std::optional<ArgAttrKind> getArgAttrKind(StringRef name);

/// Check if an attribute is present on a function argument. The argument must
/// have a parent function.
bool hasAttr(const Argument &a, ArgAttrKind attr);

/// Add an attribute to a function argument. Only attributes that do not take
/// any values can be added this way. Adding an attribute that takes values will
/// result in a catastrophic runtime error. The argument must have a parent
/// function.
void addAttr(Argument &a, ArgAttrKind attr);

/// Remove an attribute from a function argument. If the argument does not
/// contain the attribute, this has no effect. The argument must have a parent
/// function.
void removeAttr(Argument &a, ArgAttrKind attr);

/// @}

#define ARG_ATTR(...) DECL_ATTR_COMMON(Argument, __VA_ARGS__)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"

#define ARG_ATTR_L(...) DECL_ATTR_L(Argument, __VA_ARGS__)
#define ARG_ATTR_S(...) DECL_ATTR_S(Argument, __VA_ARGS__)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"

#define ARG_ATTR_0(...) DECL_ATTR_0(Argument, __VA_ARGS__)
#define ARG_ATTR_1(...) DECL_ATTR_1(Argument, __VA_ARGS__)
#define ARG_ATTR_2(...) DECL_ATTR_2(Argument, __VA_ARGS__)
#define ARG_ATTR_3(...) DECL_ATTR_3(Argument, __VA_ARGS__)
#define ARG_ATTR_4(...) DECL_ATTR_4(Argument, __VA_ARGS__)
#define ARG_ATTR_5(...) DECL_ATTR_5(Argument, __VA_ARGS__)
#define ARG_ATTR_6(...) DECL_ATTR_6(Argument, __VA_ARGS__)
#define ARG_ATTR_7(...) DECL_ATTR_7(Argument, __VA_ARGS__)
#define ARG_ATTR_8(...) DECL_ATTR_8(Argument, __VA_ARGS__)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"

#define ARG_ATTR_N(...) DECL_ATTR_N(Argument, __VA_ARGS__)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"

} // namespace llvm

#endif // KITSUNE_CORE_ARG_ATTRS_H

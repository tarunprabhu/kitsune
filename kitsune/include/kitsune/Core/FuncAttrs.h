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

#include "kitsune/Core/AttrsInternal.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class Function;
class MDNode;
class raw_ostream;

/// \addtogroup kitsune
/// @{

/// Kitsune-specific attributes for functions.
enum class FuncAttrKind : uint32_t {
#define GET_FUNC_ATTR_ENUMS
#include "kitsune/Core/FuncAttrs.inc"
};

/// Get the metadata node containing the list of Kitsune-specific attributes.
/// If no Kitsune-specific attributes have been attached to the function, this
/// may return nullptr.
MDNode *getAttrList(const Function &f);

/// Get the name of the function attribute as it appears in the LLVM-IR
/// metadata. The result will start with "kit.func.".
StringRef getAttrName(FuncAttrKind attr);

/// Get the kind of a Kitsune-specific attribute if the given string is how such
/// an attribute would appear in LLVM-IR. Otherwise, return std::nullopt.
std::optional<FuncAttrKind> getFuncAttrKind(StringRef name);

/// Check if the given attribute is present on a function.
bool hasAttr(const Function &f, FuncAttrKind attr);

/// Add an attribute to the function. Only attributes that do not take any
/// values can be added this way. Providing an attribute that takes values will
/// result in a catastrophic runtime error.
void addAttr(Function &f, FuncAttrKind attr);

/// Remove the attribute from a function. If the function does not contain the
/// attribute, this has no effect.
void removeAttr(Function &f, FuncAttrKind attr);

/// If the attribute is not present on a function, return true. Otherwise,
/// return if the expected number of values are found for the attribute, and
/// each of them can be retrieved. In all other cases, return false. If an
/// output stream is provided, an error message will be printed to it if the
/// attribute is invalid.
bool verifyAttr(const Function &f, FuncAttrKind attr,
                raw_ostream *os = nullptr);

/// @}

#define FUNC_ATTR(...) DECL_ATTR_COMMON(Function, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

#define FUNC_ATTR_LOOP(...) DECL_ATTR_LOOP(Function, __VA_ARGS__)
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

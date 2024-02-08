//===- ModuleAttrs.h - Module attributes and utilities ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions and utilities to work with module "attributes" (really named
// LLVM-IR metadata).
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_MODULE_ATTRS_H
#define KITSUNE_CORE_MODULE_ATTRS_H

#include "kitsune/Core/AttrsDeclMacros.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class Module;
class MDNode;

/// \addtogroup kitsune
/// @{

/// Kitsune-specific attributes for modules.
enum class ModuleAttrKind : uint32_t {
#define GET_MODULE_ATTR_ENUMS
#include "kitsune/Core/ModuleAttrs.inc"
};

/// Get the name of a module attribute as it would appear in LLVM metadata.
/// The result will start with "kit.module.".
StringRef getAttrName(ModuleAttrKind attrKind);

/// Get the kind of a module attribute if the given string is how the attribute
/// would appear in LLVM metadata. Otherwise, return std::nullopt.
std::optional<ModuleAttrKind> getModuleAttrKind(StringRef name);

/// Check if an attribute is present on a module.
bool hasAttr(const Module &m, ModuleAttrKind attr);

/// Add an attribute to a module. Only attributes that do not take any values
/// can be added this way. Adding an attribute that takes values will result in
/// a catastrophic runtime error.
void addAttr(Module &m, ModuleAttrKind attr);

/// Remove an attribute from a module. If the module does not contain the
/// attribute, this has no effect.
void removeAttr(Module &m, ModuleAttrKind attr);

/// @}

#define MODULE_ATTR(...) DECL_ATTR_COMMON(Module, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

#define MODULE_ATTR_0(...) DECL_ATTR_0(Module, __VA_ARGS__)
#define MODULE_ATTR_1(...) DECL_ATTR_1(Module, __VA_ARGS__)
#define MODULE_ATTR_2(...) DECL_ATTR_2(Module, __VA_ARGS__)
#define MODULE_ATTR_3(...) DECL_ATTR_3(Module, __VA_ARGS__)
#define MODULE_ATTR_4(...) DECL_ATTR_4(Module, __VA_ARGS__)
#define MODULE_ATTR_5(...) DECL_ATTR_5(Module, __VA_ARGS__)
#define MODULE_ATTR_6(...) DECL_ATTR_6(Module, __VA_ARGS__)
#define MODULE_ATTR_7(...) DECL_ATTR_7(Module, __VA_ARGS__)
#define MODULE_ATTR_8(...) DECL_ATTR_8(Module, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

#define MODULE_ATTR_N(...) DECL_ATTR_N(Module, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

} // namespace llvm

#endif // KITSUNE_CORE_MODULE_ATTRS_H

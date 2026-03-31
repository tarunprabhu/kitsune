//===- InstAttrs.h - Instruction attributes and utilities ------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions and utilities to work with "attributes" (really LLVM-IR metadata)
// on instructions.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_INST_ATTRS_H
#define KITSUNE_CORE_INST_ATTRS_H

#include "kitsune/Core/AttrsDeclMacros.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class Instruction;

/// \addtogroup kitsune
/// @{

/// Kitsune-specific attributes for instructions.
enum class InstAttrKind : uint32_t {
#define GET_INST_ATTR_ENUMS
#include "kitsune/Core/InstAttrs.inc"
};

/// Get the name of an instruction attribute as it appears in LLVM metadata.
/// The result will start with "kit.inst.".
StringRef getAttrName(InstAttrKind attr);

/// Get the kind of an instruction attribute if the given string is how the
/// attribute would appear in LLVM metadata. Otherwise, return std::nullopt.
std::optional<InstAttrKind> getInstAttrKind(StringRef name);

/// Check if an attribute is present on an instruction.
bool hasAttr(const Instruction &inst, InstAttrKind attr);

/// Add an attribute to an instruction. Only attributes that do not take any
/// values can be added this way. Adding an attribute that takes values will
/// result in a catastrophic runtime error.
void addAttr(Instruction &inst, InstAttrKind attr);

/// Remove an attribute from an instruction. If the instruction does not contain
/// the attribute, this has no effect.
void removeAttr(Instruction &inst, InstAttrKind attr);

/// @}

#define INST_ATTR(...) DECL_ATTR_COMMON(Instruction, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_LOOP(...) DECL_ATTR_LOOP(Instruction, __VA_ARGS__)
#define INST_ATTR_0(...) DECL_ATTR_0(Instruction, __VA_ARGS__)
#define INST_ATTR_1(...) DECL_ATTR_1(Instruction, __VA_ARGS__)
#define INST_ATTR_2(...) DECL_ATTR_2(Instruction, __VA_ARGS__)
#define INST_ATTR_3(...) DECL_ATTR_3(Instruction, __VA_ARGS__)
#define INST_ATTR_4(...) DECL_ATTR_4(Instruction, __VA_ARGS__)
#define INST_ATTR_5(...) DECL_ATTR_5(Instruction, __VA_ARGS__)
#define INST_ATTR_6(...) DECL_ATTR_6(Instruction, __VA_ARGS__)
#define INST_ATTR_7(...) DECL_ATTR_7(Instruction, __VA_ARGS__)
#define INST_ATTR_8(...) DECL_ATTR_8(Instruction, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_N(...) DECL_ATTR_N(Instruction, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

} // namespace llvm

#endif // KITSUNE_CORE_INST_ATTRS_H

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

#include "kitsune/Core/AttrsInternal.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class Instruction;
class MDNode;
class raw_ostream;

/// \addtogroup kitsune
/// @{

/// Kitsune-specific attributes for instructions.
enum class InstAttrKind : uint32_t {
#define GET_INST_ATTR_ENUMS
#include "kitsune/Core/InstAttrs.inc"
};

/// Get the metadata node containing the list of Kitsune-specific attributes.
/// If no Kitsune-specific attributes have been added to the instruction, this
/// may return nullptr.
MDNode *getAttrList(const Instruction &inst);

/// Get the name of the instruction attribute as it appears in the LLVM-IR
/// metadata. The result will start with "kit.inst.".
StringRef getAttrName(InstAttrKind attr);

/// Get the kind of a Kitsune-specific instruction attribute if the given string
/// is how such an attribute would appear in LLVM-IR. Otherwise, return
/// std::nullopt.
std::optional<InstAttrKind> getInstAttrKind(StringRef name);

/// Check if the given attribute is present on an instruction.
bool hasAttr(const Instruction &inst, InstAttrKind attr);

/// Add an attribute to the instruction. Only attributes that do not take any
/// values can be added this way. Providing an attribute that takes values will
/// result in a catastrophic runtime error.
void addAttr(Instruction &inst, InstAttrKind attr);

/// Remove the attribute from an instructoin. If the instruction does not
/// contain the attribute, this has no effect.
void removeAttr(Instruction &inst, InstAttrKind attr);

/// If the attribute is not present on an instruction, return true. Otherwise,
/// return if the expected number of values are found for the attribute, and
/// each of them can be retrieved. In all other cases, return false. If an
/// output stream is provided, an error message will be printed to it if the
/// attribute is invalid.
bool verifyAttr(const Instruction &inst, InstAttrKind attr,
                raw_ostream *os = nullptr);

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

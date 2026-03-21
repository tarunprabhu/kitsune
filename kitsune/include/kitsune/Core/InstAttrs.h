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

#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class Instruction;
class Loop;
class LoopInfo;
class MDNode;

/// \addtogroup kitsune
/// \@{

/// Attributes for instruction.
enum class InstAttrKind : uint32_t {
#define GET_INST_ATTR_ENUMS
#include "kitsune/Core/InstAttrs.inc"
};

/// Get the name of the instruction attribute as it appears in the LLVM-IR
/// metadata. The result will start with "kit.inst.".
StringRef getAttrName(InstAttrKind attr);

/// Get the kind of an instruction attribute if the given string corresponds to
/// the name of an attribute as it might appear in LLVM-IR. If the string does
/// not correspond to a valid attribute name, return std::nullopt.
std::optional<InstAttrKind> getInstAttrKind(StringRef name);

/// Check if the given attribute is present on an instruction.
bool hasAttr(const Instruction &inst, InstAttrKind attr);

/// Add an attribute to the instruction. Only attributes that do not take any
/// values can be added using this function. Adding any other attribute will
/// result in a catastrophic runtime error.
void addAttr(Instruction &inst, InstAttrKind attr);

/// Remove the attribute from an instructoin. If the instruction does not
/// contain the attribute, this has no effect.
void removeAttr(Instruction &inst, InstAttrKind attr);

/// @}

// Flag attributes (those that do not have a value), and attributes that take
// Loop's as values will have a different set of accessors. Mask these before
// generating declarations for the other attributes.
#define INST_ATTR_FLAG(NAME, IRNAME)
#define INST_ATTR_LOOP(NAME, IRNAME)
#define INST_ATTR(NAME, TYPE, IRNAME)                                          \
  bool has##NAME##Attr(const Instruction &inst);                               \
  std::optional<TYPE> get##NAME##Attr(const Instruction &inst);                \
  void add##NAME##Attr(Instruction &inst, TYPE val);                           \
  void remove##NAME##Attr(Instruction &inst);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_FLAG(NAME, IRNAME)                                           \
  bool has##NAME##Attr(const Instruction &inst);                               \
  void add##NAME##Attr(Instruction &inst);                                     \
  void remove##NAME##Attr(Instruction &inst);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_LOOP(NAME, IRNAME)                                           \
  bool has##NAME##Attr(const Instruction &inst);                               \
  std::optional<Loop *> get##NAME##Attr(                                       \
      const Instruction &inst, const SmallVectorImpl<const LoopInfo *> &lis);  \
  void add##NAME##Attr(Instruction &inst, const Loop *val);                    \
  void add##NAME##Attr(Instruction &inst, const Loop &val);                    \
  void remove##NAME##Attr(Instruction &inst);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

} // namespace llvm

#endif // KITSUNE_CORE_INST_ATTRS_H

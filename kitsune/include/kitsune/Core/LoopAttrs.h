//===- LoopAttrs.h - Loop attributes and utilities -------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions and utilities to work with "attributes" (really LLVM-IR metadata)
// on loops. Most of these will be for tapir loops, but, in principle,
// Kitsune-specific attributes could be applied to regular loops as well.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_LOOP_ATTRS_H
#define KITSUNE_CORE_LOOP_ATTRS_H

#include "kitsune/Core/AttrsInternal.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class Loop;
class MDNode;
class raw_ostream;

/// \addtogroup kitsune
/// @{

/// The required prefix on the names of loop metadata specific to tapir loops.
static constexpr StringRef tapirLoopAttrNamePrefix = "tapir.loop.";

/// Attributes for loops.
enum class LoopAttrKind : uint32_t {
#define GET_LOOP_ATTR_ENUMS
#include "kitsune/Core/LoopAttrs.inc"
};

/// Get the metadata node containing the list of Kitsune-specific attributes.
/// If no Kitsune-specific attributes have been added to the loop, this may
/// return nullptr.
MDNode *getAttrList(const Loop &loop);

/// Get the name of the loop attribute as it appears in the loop metadata.
/// The result will start with "tapir.loop." or "loop."
StringRef getAttrName(LoopAttrKind attr);

/// Get the kind of a Kitsune-specific loop attribute if the given string is how
/// such an attribute would appear in LLVM-IR. Otherwise, return std::nullopt.
std::optional<LoopAttrKind> getLoopAttrKind(StringRef name);

/// Check if the given attribute is present on a loop.
bool hasAttr(const Loop &loop, LoopAttrKind attr);

/// Add an attribute to the loop. Only attributes that do not take any values
/// can be added this way. Providing an attribute that takes values will result
/// in a catastrophic runtime error.
void addAttr(Loop &loop, LoopAttrKind attr);

/// Remove the attribute from a loop. If the loop does not contain the
/// attribute, this has no effect.
void removeAttr(Loop &loop, LoopAttrKind attr);

/// If the attribute is not present on a loop, return true. Otherwise, return if
/// the expected number of values are found for the attribute, and each of them
/// can be retrieved. In all other cases, return false. If an output stream is
/// provided, an error message will be printed to it if the attribute is
/// invalid.
bool verifyAttr(const Loop &loop, LoopAttrKind attr, raw_ostream *os = nullptr);

/// @}

#define LOOP_ATTR(...) DECL_ATTR_COMMON(Loop, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

#define LOOP_ATTR_LOOP(...) DECL_ATTR_LOOP(Loop, __VA_ARGS__)
#define LOOP_ATTR_0(...) DECL_ATTR_0(Loop, __VA_ARGS__)
#define LOOP_ATTR_1(...) DECL_ATTR_1(Loop, __VA_ARGS__)
#define LOOP_ATTR_2(...) DECL_ATTR_2(Loop, __VA_ARGS__)
#define LOOP_ATTR_3(...) DECL_ATTR_3(Loop, __VA_ARGS__)
#define LOOP_ATTR_4(...) DECL_ATTR_4(Loop, __VA_ARGS__)
#define LOOP_ATTR_5(...) DECL_ATTR_5(Loop, __VA_ARGS__)
#define LOOP_ATTR_6(...) DECL_ATTR_6(Loop, __VA_ARGS__)
#define LOOP_ATTR_7(...) DECL_ATTR_7(Loop, __VA_ARGS__)
#define LOOP_ATTR_8(...) DECL_ATTR_8(Loop, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

#define LOOP_ATTR_N(...) DECL_ATTR_N(Loop, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

} // namespace llvm

#endif // KITSUNE_CORE_LOOP_ATTRS_H

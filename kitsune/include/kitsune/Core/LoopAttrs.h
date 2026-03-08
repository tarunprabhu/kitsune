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

#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

/// \addtogroup kitsune
/// @{

class LLVMContext;
class Loop;
class MDNode;

/// The required prefix on the names of loop metadata specific to tapir loops.
static constexpr StringRef tapirLoopAttrNamePrefix = "tapir.loop.";

/// Attributes for loops.
enum class LoopAttrKind : uint32_t {
#define GET_LOOP_ATTR_ENUMS
#include "kitsune/Core/LoopAttrs.inc"
};

/// Get a metadata node for a loop attribute that takes a value.
template <typename T>
MDNode *getMetadataForAttr(LLVMContext &ctx, LoopAttrKind attr, T val);

/// Get a metadata node for a loop attribute that does not take a value.
MDNode *getMetadataForAttr(LLVMContext &ctx, LoopAttrKind attr);

/// Get the name of the loop attribute as it appears in the loop metadata.
/// The result will start with "tapir.loop." or "loop."
StringRef getAttrName(LoopAttrKind attr);

/// Get the kind of a loop attribute if the given string corresponds to the name
/// of an attribute as it might appear in loop metadata. If the string does not
/// correspond to a valid attribute name, return std::nullopt.
std::optional<LoopAttrKind> getLoopAttrKind(StringRef name);

/// Return true if the attribute can only be applied to a tapir loop, false
/// otherwise.
bool isAttrTapirOnly(LoopAttrKind attr);

/// Check if the given attribute is present on a loop.
bool hasAttr(const Loop &loop, LoopAttrKind attr);

/// Remove the attribute from a loop. If the loop does not contain the
/// attribute, this has no effect.
void removeAttr(Loop &loop, LoopAttrKind attr);

/// @}

// Flag attributes (those that do not have a value) will have a different set of
// accessors. Mask them by defining LOOP_ATTRIBUTE_FLAG without a body. These
// accessors are for attributes that may be applied to both tapir loops and
// regular loops.
#define LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)                                  \
  bool has##NAME##Attr(const Loop &);                                          \
  std::optional<TYPE> get##NAME##Attr(const Loop &);                           \
  void add##NAME##Attr(Loop &loop, TYPE val);                                  \
  void remove##NAME##Attr(Loop &loop);

#define LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

// Flag attributes (those that do not have a value) have a different set of
// accessors from non-flag attributes. These accessors are for attributes that
// may be applied to both tapir loops and regular loops.
#define LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                      \
  bool has##NAME##Attr(const Loop &);                                          \
  void add##NAME##Attr(Loop &loop);                                            \
  void remove##NAME##Attr(Loop &loop);

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

// Flag attributes (those that do not have a value) will have a different set of
// accessors. Mask them by defining LOOP_ATTRIBUTE_FLAG without a body. These
// accessors are for attributes that may be applied to tapir loops only.
#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)
#define TAPIR_LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)                            \
  bool has##NAME##Attr(const Loop &);                                          \
  std::optional<TYPE> get##NAME##Attr(const Loop &);                           \
  void add##NAME##Attr(Loop &loop, TYPE val);                                  \
  void remove##NAME##Attr(Loop &loop);

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

// Flag attributes (those that do not have a value) have a different set of
// accessors from non-flag attributes. These accessors are for attributes that
// may be applied to tapir loops only.
#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                \
  bool has##NAME##Attr(const Loop &);                                          \
  void add##NAME##Attr(Loop &loop);                                            \
  void remove##NAME##Attr(Loop &loop);

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

} // namespace llvm

#endif // KITSUNE_CORE_LOOP_ATTRS_H

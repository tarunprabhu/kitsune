//===- TapirLoopAttrs.h - Tapir loop attributes and utilities --*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions and utilities to work with tapir-specific attributes (really
// LLVM-IR metadata) on tapir loops.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_TAPIR_LOOP_ATTRS_H
#define KITSUNE_CORE_TAPIR_LOOP_ATTRS_H

#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

/// \addtogroup kitsune
/// @{

class LLVMContext;
class Loop;
class MDNode;

/// The required prefix on the names of all tapir loop metadata.
static constexpr StringRef tapirLoopAttrNamePrefix = "tapir.loop.";

/// Attributes for tapir loops. These will be attached as tapir loop metadata.
#define TAPIR_LOOP_ATTR(NAME) NAME,
#define TAPIR_LOOP_ATTRIBUTE_ENUM(NAME, IRNAME, TYPE) TAPIR_LOOP_ATTR(NAME)
#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME) TAPIR_LOOP_ATTR(NAME)
#define TAPIR_LOOP_ATTRIBUTE_STR(NAME, IRNAME) TAPIR_LOOP_ATTR(NAME)
#define TAPIR_LOOP_ATTRIBUTE_UINT(NAME, IRNAME) TAPIR_LOOP_ATTR(NAME)
#define TAPIR_LOOP_ATTRIBUTE_ULONG(NAME, IRNAME) TAPIR_LOOP_ATTR(NAME)

/// Attributes for tapir loops. These will be attached as tapir loop metadata.
enum class TapirLoopAttrKind : uint32_t {
#include "kitsune/Core/TapirLoopAttrs.inc"
};

#undef TAPIR_LOOP_ATTRIBUTE_ULONG
#undef TAPIR_LOOP_ATTRIBUTE_UINT
#undef TAPIR_LOOP_ATTRIBUTE_STR
#undef TAPIR_LOOP_ATTRIBUTE_FLAG
#undef TAPIR_LOOP_ATTRIBUTE_ENUM
#undef TAPIR_LOOP_ATTR

/// Get a metadata node for the tapir loop attribute that takes a value.
template <typename T>
MDNode *getMetadataForTapirLoopAttr(LLVMContext &ctx, TapirLoopAttrKind attr,
                                    T val);

/// Get a metadata node for the tapir loop attribute that does not take a value.
MDNode *getMetadataForTapirLoopAttr(LLVMContext &ctx, TapirLoopAttrKind attr);

/// Get the name of the tapir loop attribute as it appears in the loop
/// metadata.
StringRef getTapirLoopAttrName(TapirLoopAttrKind attr);

/// Check if the given attribute is present on a loop.
bool hasTapirLoopAttr(const Loop &loop, TapirLoopAttrKind attr);

/// Remove the attribute from a loop. If the loop does not contain the
/// attribute, this has no effect.
void removeTapirLoopAttr(Loop &loop, TapirLoopAttrKind attr);

/// Remove all tapir loop attributes from the loop. Returns true if at least one
/// attribute was removed, false otherwise.
void clearTapirLoopAttrs(Loop &loop);

#define TAPIR_LOOP_ATTR(NAME, TYPE)                                            \
  bool hasTapirLoop##NAME##Attr(const Loop &);                                 \
  std::optional<TYPE> getTapirLoop##NAME##Attr(const Loop &);                  \
  void addTapirLoop##NAME##Attr(Loop &loop, TYPE val);                         \
  void removeTapirLoop##NAME##Attr(Loop &loop);

#define TAPIR_LOOP_ATTRIBUTE_ENUM(NAME, IRNAME, TYPE)                          \
  TAPIR_LOOP_ATTR(NAME, TYPE)

#define TAPIR_LOOP_ATTRIBUTE_STR(NAME, IRNAME) TAPIR_LOOP_ATTR(NAME, StringRef)

#define TAPIR_LOOP_ATTRIBUTE_UINT(NAME, IRNAME) TAPIR_LOOP_ATTR(NAME, unsigned)

#define TAPIR_LOOP_ATTRIBUTE_ULONG(NAME, IRNAME)                               \
  TAPIR_LOOP_ATTR(NAME, unsigned long)

#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                \
  bool hasTapirLoop##NAME##Attr(const Loop &);                                 \
  void addTapirLoop##NAME##Attr(Loop &loop);                                   \
  void removeTapirLoop##NAME##Attr(Loop &loop);

#include "kitsune/Core/TapirLoopAttrs.inc"

#undef TAPIR_LOOP_ATTRIBUTE_ULONG
#undef TAPIR_LOOP_ATTRIBUTE_UINT
#undef TAPIR_LOOP_ATTRIBUTE_STR
#undef TAPIR_LOOP_ATTRIBUTE_FLAG
#undef TAPIR_LOOP_ATTRIBUTE_ENUM
#undef TAPIR_LOOP_ATTR

} // namespace llvm

#endif // KITSUNE_CORE_TAPIR_LOOP_ATTRS_H

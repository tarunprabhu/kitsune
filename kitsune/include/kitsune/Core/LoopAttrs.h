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

#include "kitsune/Core/MetadataUtils.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Metadata.h"

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
/// can be retrieved. In all other cases, return false.
bool verifyAttr(const Loop &loop, LoopAttrKind attr);

/// Get a metadata node for a loop attribute that takes a value.
template <typename T>
MDNode *getMDNodeForAttr(LLVMContext &ctx, LoopAttrKind attr, T val) {
  StringRef name = getAttrName(attr);
  Metadata *mdVal = toMetadata(val, ctx);
  Metadata *mdTag = MDString::get(ctx, name);
  MDNode *md = MDNode::get(ctx, {mdTag, mdVal});

  return md;
}

/// @}

#define LOOP_ATTR(NAME, IRNAME, TYPE)                                          \
  bool verify##NAME##Attr(const Loop &loop);                                   \
  bool has##NAME##Attr(const Loop &loop);                                      \
  void remove##NAME##Attr(Loop &loop);
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

#define LOOP_ATTR_0(NAME, IRNAME) void add##NAME##Attr(Loop &loop);

#define LOOP_ATTR_1(NAME, IRNAME, TYPE)                                        \
  std::optional<TYPE> get##NAME##Attr(const Loop &loop);                       \
  void add##NAME##Attr(Loop &loop, TYPE val);

#define LOOP_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)        \
  void add##NAME##Attr(Loop &loop, ETY0 e0, ETY1 e1);

#define LOOP_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2)                                               \
  void add##NAME##Attr(Loop &loop, ETY0 e0, ETY1 e1, ETY2 e2);

#define LOOP_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3)                            \
  void add##NAME##Attr(Loop &loop, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3);

#define LOOP_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4)         \
  void add##NAME##Attr(Loop &loop, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3, ETY4 e4);

#define LOOP_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5)                                               \
  void add##NAME##Attr(Loop &loop, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,         \
                       ETY4 e4, ETY5 en5);

#define LOOP_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6)                            \
  void add##NAME##Attr(Loop &loop, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,         \
                       ETY4 e4, ETY5 en5, ETY6 en6);

#define LOOP_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7)         \
  void add##NAME##Attr(Loop &loop, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,         \
                       ETY4 e4, ETY5 en5, ETY6 en6, ETY7 en7);
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

#define LOOP_ATTR_N(NAME, IRNAME, ETY, ENAME, EN, NELEMS)                      \
  std::optional<ETY> get##ENAME##From##NAME##Attr(const Loop &loop);
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

} // namespace llvm

#endif // KITSUNE_CORE_LOOP_ATTRS_H

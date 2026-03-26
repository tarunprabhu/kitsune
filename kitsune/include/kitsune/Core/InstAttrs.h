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

#define INST_ATTR(NAME, IRNAME, TYPE)                                          \
  bool verify##NAME##Attr(const Instruction &inst, raw_ostream *os = nullptr); \
  bool has##NAME##Attr(const Instruction &inst);                               \
  void remove##NAME##Attr(Instruction &inst);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_LOOP(NAME, IRNAME)                                           \
  std::optional<Loop *> get##NAME##Attr(                                       \
      const Instruction &inst, const SmallVectorImpl<const LoopInfo *> &lis);  \
  void add##NAME##Attr(Instruction &inst, const Loop &loop);

#define INST_ATTR_0(NAME, IRNAME) void add##NAME##Attr(Instruction &inst);

#define INST_ATTR_1(NAME, IRNAME, TYPE)                                        \
  std::optional<TYPE> get##NAME##Attr(const Instruction &inst);                \
  void add##NAME##Attr(Instruction &inst, TYPE val);

#define INST_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)        \
  void add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1);

#define INST_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2)                                               \
  void add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1, ETY2 e2);

#define INST_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3)                            \
  void add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3);

#define INST_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4)         \
  void add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,  \
                       ETY4 e4);

#define INST_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5)                                               \
  void add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,  \
                       ETY4 e4, ETY5 en5);

#define INST_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6)                            \
  void add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,  \
                       ETY4 e4, ETY5 en5, ETY6 en6);

#define INST_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7)         \
  void add##NAME##Attr(Instruction &inst, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,  \
                       ETY4 e4, ETY5 en5, ETY6 en6, ETY7 en7);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_N(NAME, IRNAME, ETY, ENAME, EN, NELEMS)                      \
  std::optional<ETY> get##ENAME##From##NAME##Attr(const Instruction &inst);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

} // namespace llvm

#endif // KITSUNE_CORE_INST_ATTRS_H

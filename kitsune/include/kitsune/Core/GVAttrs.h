//===- GVAttrs.h - Kitsune-specific attributes for global vars -*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions and utilities to work with Kitsune-specific attributes for global
// variables.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_GV_ATTRS_H
#define KITSUNE_CORE_GV_ATTRS_H

#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class GlobalVariable;

/// \addtogroup kitsune
/// \@{

/// Kitsune-specific attributes for global variables.
enum class GVAttrKind : uint32_t {
#define GET_GV_ATTR_ENUMS
#include "kitsune/Core/GVAttrs.inc"
};

/// Get the name of the global variable attribute as it appears in the LLVM-IR
/// metadata. The result will start with "kit.gv.".
StringRef getAttrName(GVAttrKind attr);

/// Get the kind of a Kitsune-specific global variable attribute if the given
/// string is how such an attribute would appear in LLVM-IR. Otherwise, return
/// std::nullopt.
std::optional<GVAttrKind> getGVAttrKind(StringRef name);

/// Check if the given attribute is present on a global variable.
bool hasAttr(const GlobalVariable &f, GVAttrKind attr);

/// Add an attribute to the global variable. Only attributes that do not take
/// any values can be added this way. Providing an attribute that takes values
/// will result in a catastrophic runtime error.
void addAttr(GlobalVariable &f, GVAttrKind attr);

/// Remove the attribute from a global variable. If the global variable does not
/// contain the attribute, this has no effect.
void removeAttr(GlobalVariable &f, GVAttrKind attr);

/// @}

#define GV_ATTR(NAME, IRNAME, TYPE)                                            \
  bool has##NAME##Attr(const GlobalVariable &g);                               \
  void remove##NAME##Attr(GlobalVariable &g);
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

#define GV_ATTR_0(NAME, IRNAME) void add##NAME##Attr(GlobalVariable &g);

#define GV_ATTR_1(NAME, IRNAME, TYPE)                                          \
  std::optional<TYPE> get##NAME##Attr(const GlobalVariable &g);                \
  void add##NAME##Attr(GlobalVariable &g, TYPE val);

#define GV_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)          \
  void add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1);

#define GV_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2)                                                 \
  void add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1, ETY2 e2);

#define GV_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2, ETY3, ENAME3, EN3)                              \
  void add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3);

#define GV_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4)           \
  void add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,  \
                       ETY4 e4);

#define GV_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,     \
                  ENAME5, EN5)                                                 \
  void add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,  \
                       ETY4 e4, ETY5 en5);

#define GV_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,     \
                  ENAME5, EN5, ETY6, ENAME6, EN6)                              \
  void add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,  \
                       ETY4 e4, ETY5 en5, ETY6 en6);

#define GV_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,     \
                  ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7)           \
  void add##NAME##Attr(GlobalVariable &g, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,  \
                       ETY4 e4, ETY5 en5, ETY6 en6, ETY7 en7);
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

#define GV_ATTR_N(NAME, IRNAME, ETY, ENAME, EN, NELEMS)                        \
  std::optional<ETY> get##ENAME##From##NAME##Attr(const GlobalVariable &g);
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

} // namespace llvm

#endif // KITSUNE_CORE_GV_ATTRS_H

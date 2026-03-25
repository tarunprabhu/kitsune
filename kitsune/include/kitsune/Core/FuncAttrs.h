//===- FuncAttrs.h - Kitsune-specific attributes for functions -*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions and utilities to work with function attributes. These are not
// known to LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_FUNC_ATTRS_H
#define KITSUNE_CORE_FUNC_ATTRS_H

#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class Function;
class raw_ostream;

/// \addtogroup kitsune
/// @{

/// Kitsune-specific attributes for functions.
enum class FuncAttrKind : uint32_t {
#define GET_FUNC_ATTR_ENUMS
#include "kitsune/Core/FuncAttrs.inc"
};

/// Get the name of the function attribute as it appears in the LLVM-IR
/// metadata. The result will start with "kit.func.".
StringRef getAttrName(FuncAttrKind attr);

/// Get the kind of a Kitsune-specific attribute if the given string is how such
/// an attribute would appear in LLVM-IR. Otherwise, return std::nullopt.
std::optional<FuncAttrKind> getFuncAttrKind(StringRef name);

/// Check if the given attribute is present on a function.
bool hasAttr(const Function &f, FuncAttrKind attr);

/// Add an attribute to the function. Only attributes that do not take any
/// values can be added this way. Providing an attribute that takes values will
/// result in a catastrophic runtime error.
void addAttr(Function &f, FuncAttrKind attr);

/// Remove the attribute from a function. If the function does not contain the
/// attribute, this has no effect.
void removeAttr(Function &f, FuncAttrKind attr);

/// If the attribute is not present on a function, return true. Otherwise,
/// return if the expected number of values are found for the attribute, and
/// each of them can be retrieved. In all other cases, return false. If an
/// output stream is provided, an error message will be printed to it if the
/// attribute is invalid.
bool verifyAttr(const Function &f, FuncAttrKind attr,
                raw_ostream *os = nullptr);

/// @}

#define FUNC_ATTR(NAME, IRNAME, TYPE)                                          \
  bool verify##NAME##Attr(const Function &f, raw_ostream *os = nullptr);       \
  bool has##NAME##Attr(const Function &f);                                     \
  void remove##NAME##Attr(Function &f);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

#define FUNC_ATTR_0(NAME, IRNAME) void add##NAME##Attr(Function &f);

#define FUNC_ATTR_1(NAME, IRNAME, TYPE)                                        \
  std::optional<TYPE> get##NAME##Attr(const Function &f);                      \
  void add##NAME##Attr(Function &f, TYPE val);

#define FUNC_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)        \
  void add##NAME##Attr(Function &f, ETY0 e0, ETY1 e1);

#define FUNC_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2)                                               \
  void add##NAME##Attr(Function &f, ETY0 e0, ETY1 e1, ETY2 e2);

#define FUNC_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3)                            \
  void add##NAME##Attr(Function &f, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3);

#define FUNC_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4)         \
  void add##NAME##Attr(Function &f, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,        \
                       ETY4 e4);

#define FUNC_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5)                                               \
  void add##NAME##Attr(Function &f, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,        \
                       ETY4 e4, ETY5 en5);

#define FUNC_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6)                            \
  void add##NAME##Attr(Function &f, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,        \
                       ETY4 e4, ETY5 en5, ETY6 en6);

#define FUNC_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7)         \
  void add##NAME##Attr(Function &f, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3,        \
                       ETY4 e4, ETY5 en5, ETY6 en6, ETY7 en7);

#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

#define FUNC_ATTR_N(NAME, IRNAME, ETY, ENAME, EN, NELEMS)                      \
  std::optional<ETY> get##ENAME##From##NAME##Attr(const Function &f);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

} // namespace llvm

#endif // KITSUNE_CORE_FUNC_ATTRS_H

//===- ModuleAttrs.h - Module attributes and utilities ---------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions and utilities to work with module "attributes" (really named
// LLVM-IR metadata).
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_MODULE_ATTRS_H
#define KITSUNE_CORE_MODULE_ATTRS_H

#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

/// \addtogroup kitsune
/// @{

class Loop;
class LoopInfo;
class Module;
class MDNode;
class raw_ostream;

/// Kitsune-specific attributes for modules.
enum class ModuleAttrKind : uint32_t {
#define GET_MODULE_ATTR_ENUMS
#include "kitsune/Core/ModuleAttrs.inc"
};

/// Get the metadata node containing the list of Kitsune-specific attributes.
/// If no Kitsune-specific attributes have been attached to the module, this
/// may return nullptr.
MDNode *getAttrList(const Module &m);

/// Get the name of the module attribute as it appears in the metadata. The
/// result will start with "kit.module.".
StringRef getAttrName(ModuleAttrKind attrKind);

/// Get the kind of a Kitsune-specific module attribute if the given string is
/// how such an attribute would appear in LLVM-IR. Otherwise, return
/// std::nullopt.
std::optional<ModuleAttrKind> getModuleAttrKind(StringRef name);

/// Check if the given attribute is present in a module.
bool hasAttr(const Module &m, ModuleAttrKind attr);

/// Add an attribute to the module. Only attributes that do not take any values
/// can be added this way. Providing an attribute that takes values will result
/// in a catastrophic runtime error.
void addAttr(Module &m, ModuleAttrKind attr);

/// Remove the attribute from a module. If the module does not contain the
/// attribute, this has no effect.
void removeAttr(Module &m, ModuleAttrKind attr);

/// If the attribute is not present on a module, return true. Otherwise, return
/// if the expected number of values are found for the attribute, and each of
/// them can be retrieved. In all other cases, return false. If an output stream
/// is provided, an error message will be printed to it if the attribute is
/// invalid.
bool verifyAttr(const Module &m, ModuleAttrKind attr,
                raw_ostream *os = nullptr);

/// @}

#define MODULE_ATTR(NAME, IRNAME, TYPE)                                        \
  bool verify##NAME##Attr(const Module &m, raw_ostream *os = nullptr);         \
  bool has##NAME##Attr(const Module &m);                                       \
  void remove##NAME##Attr(Module &m);
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

#define MODULE_ATTR_LOOP(NAME, IRNAME)                                         \
  std::optional<Loop *> get##NAME##Attr(                                       \
      const Module &m, const SmallVectorImpl<const LoopInfo *> &lis);          \
  void add##NAME##Attr(Module &m, const Loop &loop);

#define MODULE_ATTR_0(NAME, IRNAME) void add##NAME##Attr(Module &m);

#define MODULE_ATTR_1(NAME, IRNAME, TYPE)                                      \
  std::optional<TYPE> get##NAME##Attr(const Module &m);                        \
  void add##NAME##Attr(Module &m, TYPE val);

#define MODULE_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)      \
  void add##NAME##Attr(Module &m, ETY0 e0, ETY1 e1);

#define MODULE_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2)                                       \
  void add##NAME##Attr(Module &m, ETY0 e0, ETY1 e1, ETY2 e2);

#define MODULE_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3)                    \
  void add##NAME##Attr(Module &m, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3);

#define MODULE_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4) \
  void add##NAME##Attr(Module &m, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3, ETY4 e4);

#define MODULE_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, \
                      ETY5, ENAME5, EN5)                                       \
  void add##NAME##Attr(Module &m, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3, ETY4 e4, \
                       ETY5 en5);

#define MODULE_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, \
                      ETY5, ENAME5, EN5, ETY6, ENAME6, EN6)                    \
  void add##NAME##Attr(Module &m, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3, ETY4 e4, \
                       ETY5 en5, ETY6 en6);

#define MODULE_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, \
                      ETY5, ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7) \
  void add##NAME##Attr(Module &m, ETY0 e0, ETY1 e1, ETY2 e2, ETY3 e3, ETY4 e4, \
                       ETY5 en5, ETY6 en6, ETY7 en7);

#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

#define MODULE_ATTR_N(NAME, IRNAME, ETY, ENAME, EN, NELEMS)                    \
  std::optional<ETY> get##ENAME##From##NAME##Attr(const Module &m);
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

} // namespace llvm

#endif // KITSUNE_CORE_MODULE_ATTRS_H

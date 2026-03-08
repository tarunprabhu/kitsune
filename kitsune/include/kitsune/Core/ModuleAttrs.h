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

class LLVMContext;
class Module;
class NamedMDNode;

/// Attributes for modules.
enum class ModuleAttrKind : uint32_t {
#define GET_MODULE_ATTR_ENUMS
#include "kitsune/Core/ModuleAttrs.inc"
};

/// Get the name of the module attribute as it appears in the metadata. The
/// result will start with "kit.module.".
StringRef getAttrName(ModuleAttrKind attrKind);

/// Get the kind of a module attribute if the given string corresponds to the
/// name of an attribute as it might appear in loop metadata. If the string does
/// not correspond to a valid attribute name, return std::nullopt.
std::optional<ModuleAttrKind> getModuleAttrKind(StringRef name);

/// Check if the given attribute is present in a module.
bool hasAttr(const Module &m, ModuleAttrKind attr);

/// Remove the attribute from a module. If the module does not contain the
/// attribute, this has no effect.
void removeAttr(Module &m, ModuleAttrKind attr);

/// @}

#define MODULE_ATTR(NAME, IRNAME)                                              \
  bool has##NAME##Attr(const Module &m);                                       \
  void remove##NAME##Attr(Module &m);

#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

#define MODULE_ATTR_0(NAME, IRNAME) NamedMDNode &add##NAME##Attr(Module &m);

#define MODULE_ATTR_1(NAME, IRNAME, TY1, V1)                                   \
  NamedMDNode &add##NAME##Attr(Module &m, TY1 V1);                             \
  std::optional<TY1> get##V1##From##NAME##Attr(const Module &m);

#define MODULE_ATTR_2(NAME, IRNAME, TY1, V1, TY2, V2)                          \
  NamedMDNode &add##NAME##Attr(Module &m, TY1 V1, TY2 V2);                     \
  std::optional<TY1> get##V1##From##NAME##Attr(const Module &m);               \
  std::optional<TY2> get##V2##From##NAME##Attr(const Module &m);

#define MODULE_ATTR_3(NAME, IRNAME, TY1, V1, TY2, V2, TY3, V3)                 \
  NamedMDNode &add##NAME##Attr(Module &m, TY1 V1, TY2 V2, TY3 V3);             \
  std::optional<TY1> get##V1##From##NAME##Attr(const Module &m);               \
  std::optional<TY2> get##V2##From##NAME##Attr(const Module &m);               \
  std::optional<TY3> get##V3##From##NAME##Attr(const Module &m);

#define MODULE_ATTR_4(NAME, IRNAME, TY1, V1, TY2, V2, TY3, V3, TY4, V4)        \
  NamedMDNode &add##NAME##Attr(Module &m, TY1 V1, TY2 V2, TY3 V3, TY4 V4);     \
  std::optional<TY1> get##V1##From##NAME##Attr(const Module &m);               \
  std::optional<TY2> get##V2##From##NAME##Attr(const Module &m);               \
  std::optional<TY3> get##V3##From##NAME##Attr(const Module &m);               \
  std::optional<TY4> get##V4##From##NAME##Attr(const Module &m);

#define MODULE_ATTR_5(NAME, IRNAME, TY1, V1, TY2, V2, TY3, V3, TY4, V4, TY5,   \
                      V5)                                                      \
  NamedMDNode &add##NAME##Attr(Module &m, TY1 V1, TY2 V2, TY3 V3, TY4 V4,      \
                               TY5 V5);                                        \
  std::optional<TY1> get##V1##From##NAME##Attr(const Module &m);               \
  std::optional<TY2> get##V2##From##NAME##Attr(const Module &m);               \
  std::optional<TY3> get##V3##From##NAME##Attr(const Module &m);               \
  std::optional<TY4> get##V4##From##NAME##Attr(const Module &m);               \
  std::optional<TY5> get##V5##From##NAME##Attr(const Module &m);

#define MODULE_ATTR_6(NAME, IRNAME, TY1, V1, TY2, V2, TY3, V3, TY4, V4, TY5,   \
                      V5, TY6, V6)                                             \
  NamedMDNode &add##NAME##Attr(Module &m, TY1 V1, TY2 V2, TY3 V3, TY4 V4,      \
                               TY5 V5, TY6 V6);                                \
  std::optional<TY1> get##V1##From##NAME##Attr(const Module &m);               \
  std::optional<TY2> get##V2##From##NAME##Attr(const Module &m);               \
  std::optional<TY3> get##V3##From##NAME##Attr(const Module &m);               \
  std::optional<TY4> get##V4##From##NAME##Attr(const Module &m);               \
  std::optional<TY5> get##V5##From##NAME##Attr(const Module &m);               \
  std::optional<TY6> get##V6##From##NAME##Attr(const Module &m);

#define MODULE_ATTR_7(NAME, IRNAME, TY1, V1, TY2, V2, TY3, V3, TY4, V4, TY5,   \
                      V5, TY6, V6, TY7, V7)                                    \
  NamedMDNode &add##NAME##Attr(Module &m, TY1 V1, TY2 V2, TY3 V3, TY4 V4,      \
                               TY5 V5, TY6 V6, TY7 V7);                        \
  std::optional<TY1> get##V1##From##NAME##Attr(const Module &m);               \
  std::optional<TY2> get##V2##From##NAME##Attr(const Module &m);               \
  std::optional<TY3> get##V3##From##NAME##Attr(const Module &m);               \
  std::optional<TY4> get##V4##From##NAME##Attr(const Module &m);               \
  std::optional<TY5> get##V5##From##NAME##Attr(const Module &m);               \
  std::optional<TY6> get##V6##From##NAME##Attr(const Module &m);               \
  std::optional<TY7> get##V7##From##NAME##Attr(const Module &m);

#define MODULE_ATTR_8(NAME, IRNAME, TY1, V1, TY2, V2, TY3, V3, TY4, V4, TY5,   \
                      V5, TY6, V6, TY7, V7, TY8, V8)                           \
  NamedMDNode &add##NAME##Attr(Module &m, TY1 V1, TY2 V2, TY3 V3, TY4 V4,      \
                               TY5 V5, TY6 V6, TY7 V7, TY8 V8);                \
  std::optional<TY1> get##V1##From##NAME##Attr(const Module &m);               \
  std::optional<TY2> get##V2##From##NAME##Attr(const Module &m);               \
  std::optional<TY3> get##V3##From##NAME##Attr(const Module &m);               \
  std::optional<TY4> get##V4##From##NAME##Attr(const Module &m);               \
  std::optional<TY5> get##V5##From##NAME##Attr(const Module &m);               \
  std::optional<TY6> get##V6##From##NAME##Attr(const Module &m);               \
  std::optional<TY7> get##V7##From##NAME##Attr(const Module &m);               \
  std::optional<TY8> get##V8##From##NAME##Attr(const Module &m);

#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

} // namespace llvm

#endif // KITSUNE_CORE_MODULE_ATTRS_H

//===- AttrsDeclMacros.h - Macros for attribute accessor decls --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Macros to generate API declarations for Kitsune-specific attributes.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_ATTRS_DECL_MACROS_H
#define KITSUNE_CORE_ATTRS_DECL_MACROS_H

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

class Loop;
class MDNode;

} // namespace llvm

#define DECL_ATTR_COMMON(IRELEM, NAME, IRNAME, CUSTOMVERIFY, TYPE)             \
  bool has##NAME##Attr(const IRELEM &ir);                                      \
  void remove##NAME##Attr(IRELEM &ir);

#define DECL_ATTR_N(IRELEM, NAME, IRNAME, ETY, ENAME, EN, NELEMS)              \
  std::optional<ETY> get##ENAME##From##NAME##Attr(const IRELEM &ir);

#define DECL_ATTR_L(IRELEM, NAME, IRNAME, CUSTOMVERIFY, TYPE)                  \
  std::optional<SmallVector<TYPE, 0>> get##NAME##Attr(const IRELEM &ir);       \
  void add##NAME##Attr(IRELEM &ir, SmallVector<TYPE, 0> const &ont);           \
  void addTo##NAME##Attr(IRELEM &ir, TYPE const &val);                         \
  void removeFrom##NAME##Attr(IRELEM &ir, TYPE const &val);

#define DECL_ATTR_S(IRELEM, NAME, IRNAME, CUSTOMVERIFY, TYPE)                  \
  std::optional<SmallSet<TYPE, 0>> get##NAME##Attr(const IRELEM &ir);          \
  void add##NAME##Attr(IRELEM &ir, SmallSet<TYPE, 0> const &cont);             \
  void addTo##NAME##Attr(IRELEM &ir, TYPE const &val);                         \
  void removeFrom##NAME##Attr(IRELEM &ir, TYPE const &val);

#define DECL_ATTR_0(IRELEM, NAME, IRNAME, CUSTOMVERIFY)                        \
  void add##NAME##Attr(IRELEM &ir);

#define DECL_ATTR_1(IRELEM, NAME, IRNAME, CUSTOMVERIFY, TYPE)                  \
  std::optional<TYPE> get##NAME##Attr(const IRELEM &ir);                       \
  void add##NAME##Attr(IRELEM &ir, TYPE const &val);

#define DECL_ATTR_2(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1)                                         \
  void add##NAME##Attr(IRELEM &ir, ETY0 const &e0, ETY1 const &e1);

#define DECL_ATTR_3(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2)                      \
  void add##NAME##Attr(IRELEM &ir, ETY0 const &e0, ETY1 const &e1,             \
                       ETY2 const &e2);

#define DECL_ATTR_4(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3)   \
  void add##NAME##Attr(IRELEM &ir, ETY0 const &e0, ETY1 const &e1,             \
                       ETY2 const &e2, ETY3 const &e3);

#define DECL_ATTR_5(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4)                                         \
  void add##NAME##Attr(IRELEM &ir, ETY0 const &e0, ETY1 const &e1,             \
                       ETY2 const &e2, ETY3 const &e3, ETY4 const &e4);

#define DECL_ATTR_6(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4, ETY5, ENAME5, EN5)                      \
  void add##NAME##Attr(IRELEM &ir, ETY0 const &e0, ETY1 const &e1,             \
                       ETY2 const &e2, ETY3 const &e3, ETY4 const &e4,         \
                       ETY5 const &e5);

#define DECL_ATTR_7(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4, ETY5, ENAME5, EN5, ETY6, ENAME6, EN6)   \
  void add##NAME##Attr(IRELEM &ir, ETY0 const &e0, ETY1 const &e1,             \
                       ETY2 const &e2, ETY3 const &e3, ETY4 const &e4,         \
                       ETY5 const &e5, ETY6 const &e6);

#define DECL_ATTR_8(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4, ETY5, ENAME5, EN5, ETY6, ENAME6, EN6,   \
                    ETY7, ENAME7, EN7)                                         \
  void add##NAME##Attr(IRELEM &ir, ETY0 const &e0, ETY1 const &e1,             \
                       ETY2 const &e2, ETY3 const &e3, ETY4 const &e4,         \
                       ETY5 const &e5, ETY6 const &e6, ETY7 const &e7);

#endif // KITSUNE_CORE_ATTRS_DECL_MACROS_H

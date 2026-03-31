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

namespace llvm {

class Loop;
class LoopInfo;

} // namespace llvm

#define DECL_ATTR_COMMON(IRELEM, NAME, IRNAME, CUSTOMVERIFY, TYPE)             \
  bool has##NAME##Attr(const IRELEM &ir);                                      \
  void remove##NAME##Attr(IRELEM &ir);

#define DECL_ATTR_LOOP(IRELEM, NAME, IRNAME, CUSTOMVERIFY)                     \
  std::optional<Loop *> get##NAME##Attr(                                       \
      const IRELEM &ir, const SmallVectorImpl<const LoopInfo *> &lis);         \
  void add##NAME##Attr(IRELEM &ir, const Loop &loop);

#define DECL_ATTR_N(IRELEM, NAME, IRNAME, ETY, ENAME, EN, NELEMS)              \
  std::optional<ETY> get##ENAME##From##NAME##Attr(const IRELEM &);

#define DECL_ATTR_0(IRELEM, NAME, IRNAME, CUSTOMVERIFY)                        \
  void add##NAME##Attr(IRELEM &ir);

#define DECL_ATTR_1(IRELEM, NAME, IRNAME, CUSTOMVERIFY, TYPE)                  \
  void add##NAME##Attr(IRELEM &ir, const TYPE &val);                           \
  std::optional<TYPE> get##NAME##Attr(const IRELEM &f);

#define DECL_ATTR_2(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1)                                         \
  void add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1);

#define DECL_ATTR_3(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2)                      \
  void add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,             \
                       const ETY2 &e2);

#define DECL_ATTR_4(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3)   \
  void add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,             \
                       const ETY2 &e2, const ETY3 &e3);

#define DECL_ATTR_5(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4)                                         \
  void add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,             \
                       const ETY2 &e2, const ETY3 &e3, const ETY4 &e4);

#define DECL_ATTR_6(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4, ETY5, ENAME5, EN5)                      \
  void add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,             \
                       const ETY2 &e2, const ETY3 &e3, const ETY4 &e4,         \
                       const ETY5 &e5);

#define DECL_ATTR_7(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4, ETY5, ENAME5, EN5, ETY6, ENAME6, EN6)   \
  void add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,             \
                       const ETY2 &e2, const ETY3 &e3, const ETY4 &e4,         \
                       const ETY5 &e5, const ETY6 &e6);

#define DECL_ATTR_8(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4, ETY5, ENAME5, EN5, ETY6, ENAME6, EN6,   \
                    ETY7, ENAME7, EN7)                                         \
  void add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,             \
                       const ETY2 &e2, const ETY3 &e3, const ETY4 &e4,         \
                       const ETY5 &e5, const ETY6 &e6, const ETY7 &e7);

#endif // KITSUNE_CORE_ATTRS_DECL_MACROS_H

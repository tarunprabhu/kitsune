//===- AttrsImplDeclMacros.h - Helper macros for attributes -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper macros to generate declarations for private accessor functions for
// Kitsune-specific attributes
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_CORE_ATTR_IMPL_DECL_MACROS_H
#define KITSUNE_LIB_CORE_ATTR_IMPL_DECL_MACROS_H

namespace llvm {

class MDNode;

class KitVerifier;

} // namespace llvm

#define DECL_ATTR_IMPL_COMMON(IRELEM, NAME, IRNAME, CUSTOMVERIFY, TYPE)        \
  void verify##NAME##Attr(KitVerifier &v, const IRELEM &ir);

#define DECL_ATTR_IMPL_0(IRELEM, NAME, IRNAME, CUSTOMVERIFY)                   \
  void verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, const bool &t);

#define DECL_ATTR_IMPL_1(IRELEM, NAME, IRNAME, CUSTOMVERIFY, TYPE)             \
  void verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, TYPE const &val);

#define DECL_ATTR_IMPL_2(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0,     \
                         EN0, ETY1, ENAME1, EN1)                               \
  void verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, ETY0 const &v0,    \
                          ETY1 const &v1);

#define DECL_ATTR_IMPL_3(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0,     \
                         EN0, ETY1, ENAME1, EN1, ETY2, ENAME2, EN2)            \
  void verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, ETY0 const &v0,    \
                          ETY1 const &v1, ETY2 const &v2);

#define DECL_ATTR_IMPL_4(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0,     \
                         EN0, ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3,      \
                         ENAME3, EN3)                                          \
  void verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, ETY0 const &v0,    \
                          ETY1 const &v1, ETY2 const &v2, ETY3 const &v3);

#define DECL_ATTR_IMPL_5(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0,     \
                         EN0, ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3,      \
                         ENAME3, EN3, ETY4, ENAME4, EN4)                       \
  void verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, ETY0 const &v0,    \
                          ETY1 const &v1, ETY2 const &v2, ETY3 const &v3,      \
                          ETY4 const &v4);

#define DECL_ATTR_IMPL_6(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0,     \
                         EN0, ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3,      \
                         ENAME3, EN3, ETY4, ENAME4, EN4, ETY5, ENAME5, EN5)    \
  void verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, ETY0 const &v0,    \
                          ETY1 const &v1, ETY2 const &v2, ETY3 const &v3,      \
                          ETY4 const &v4, ETY5 const &v5);

#define DECL_ATTR_IMPL_7(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0,     \
                         EN0, ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3,      \
                         ENAME3, EN3, ETY4, ENAME4, EN4, ETY5, ENAME5, EN5,    \
                         ETY6, ENAME6, EN6)                                    \
  void verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, ETY0 const &v0,    \
                          ETY1 const &v1, ETY2 const &v2, ETY3 const &v3,      \
                          ETY4 const &v4, ETY5 const &v5, ETY6 const &v6);

#define DECL_ATTR_IMPL_8(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0,     \
                         EN0, ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3,      \
                         ENAME3, EN3, ETY4, ENAME4, EN4, ETY5, ENAME5, EN5,    \
                         ETY6, ENAME6, EN6, ETY7, ENAME7, EN7)                 \
  void verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, ETY0 const &v0,    \
                          ETY1 const &v1, ETY2 const &v2, ETY3 const &v3,      \
                          ETY4 const &v4, ETY5 const &v5, ETY6 const &v6,      \
                          ETY7 const &v7);

#endif // KITSUNE_LIB_CORE_ATTR_IMPL_DECL_MACROS_H

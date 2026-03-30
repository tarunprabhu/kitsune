//===- AttrsInternal.h - Helper macros for Kitsune's attributes -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper macros for Kitsune-specific attributes
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_ATTRS_INTERNAL_H
#define KITSUNE_CORE_ATTRS_INTERNAL_H

#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

class Loop;
class LoopInfo;
class KitVerifier;

} // namespace llvm

#define DECL_ATTR_COMMON(IRELEM, NAME, IRNAME, CUSTOMVERIFY, TYPE)             \
  bool has##NAME##Attr(const IRELEM &ir);                                      \
  void remove##NAME##Attr(IRELEM &ir);                                         \
  bool verify##NAME##Attr(KitVerifier &v, const IRELEM &ir);

#define DECL_ATTR_LOOP(IRELEM, NAME, IRNAME, CUSTOMVERIFY)                     \
  std::optional<Loop *> get##NAME##Attr(                                       \
      const IRELEM &ir, const SmallVectorImpl<const LoopInfo *> &lis);         \
  void add##NAME##Attr(IRELEM &ir, const Loop &loop);                          \
  bool verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, const MDNode &md);

#define DECL_ATTR_N(IRELEM, NAME, IRNAME, ETY, ENAME, EN, NELEMS)              \
  std::optional<ETY> get##ENAME##From##NAME##Attr(const IRELEM &);

#define DECL_ATTR_0(IRELEM, NAME, IRNAME, CUSTOMVERIFY)                        \
  void add##NAME##Attr(IRELEM &ir);                                            \
  bool verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, const bool &t);

#define DECL_ATTR_1(IRELEM, NAME, IRNAME, CUSTOMVERIFY, TYPE)                  \
  void add##NAME##Attr(IRELEM &ir, const TYPE &val);                           \
  std::optional<TYPE> get##NAME##Attr(const IRELEM &f);                        \
  bool verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, const TYPE &val);

#define DECL_ATTR_2(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1)                                         \
  void add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1);            \
  bool verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, const ETY0 &v0,    \
                          const ETY1 &v1);

#define DECL_ATTR_3(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2)                      \
  void add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,             \
                       const ETY2 &e2);                                        \
  bool verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, const ETY0 &v0,    \
                          const ETY1 &v1, const ETY2 &v2);

#define DECL_ATTR_4(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3)   \
  void add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,             \
                       const ETY2 &e2, const ETY3 &e3);                        \
  bool verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, const ETY0 &v0,    \
                          const ETY1 &v1, const ETY2 &v2, const ETY3 &v3);

#define DECL_ATTR_5(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4)                                         \
  void add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,             \
                       const ETY2 &e2, const ETY3 &e3, const ETY4 &e4);        \
  bool verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, const ETY0 &v0,    \
                          const ETY1 &v1, const ETY2 &v2, const ETY3 &v3,      \
                          const ETY4 &v4);

#define DECL_ATTR_6(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4, ETY5, ENAME5, EN5)                      \
  void add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,             \
                       const ETY2 &e2, const ETY3 &e3, const ETY4 &e4,         \
                       const ETY5 &e5);                                        \
  bool verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, const ETY0 &v0,    \
                          const ETY1 &v1, const ETY2 &v2, const ETY3 &v3,      \
                          const ETY4 &v4, const ETY5 &v5);

#define DECL_ATTR_7(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4, ETY5, ENAME5, EN5, ETY6, ENAME6, EN6)   \
  void add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,             \
                       const ETY2 &e2, const ETY3 &e3, const ETY4 &e4,         \
                       const ETY5 &e5, const ETY6 &e6);                        \
  bool verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, const ETY0 &v0,    \
                          const ETY1 &v1, const ETY2 &v2, const ETY3 &v3,      \
                          const ETY4 &v4, const ETY5 &v5, const ETY6 &v6);

#define DECL_ATTR_8(IRELEM, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,     \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4, ETY5, ENAME5, EN5, ETY6, ENAME6, EN6,   \
                    ETY7, ENAME7, EN7)                                         \
  void add##NAME##Attr(IRELEM &ir, const ETY0 &e0, const ETY1 &e1,             \
                       const ETY2 &e2, const ETY3 &e3, const ETY4 &e4,         \
                       const ETY5 &e5, const ETY6 &e6, const ETY7 &e7);        \
  bool verify##NAME##Attr(KitVerifier &v, const IRELEM &ir, const ETY0 &v0,    \
                          const ETY1 &v1, const ETY2 &v2, const ETY3 &v3,      \
                          const ETY4 &v4, const ETY5 &v5, const ETY6 &v6,      \
                          const ETY7 &v7);

#endif // KITSUNE_CORE_ATTRS_INTERNAL_H

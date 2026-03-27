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
class raw_ostream;

} // namespace llvm

#define DECL_ATTR_COMMON(IRTYPE, NAME, IRNAME, TYPE)                           \
  bool has##NAME##Attr(const IRTYPE &);                                        \
  void remove##NAME##Attr(IRTYPE &);                                           \
  bool verify##NAME##Attr(const IRTYPE &, raw_ostream *os = nullptr);

#define DECL_ATTR_LOOP(IRTYPE, NAME, IRNAME)                                   \
  std::optional<Loop *> get##NAME##Attr(                                       \
      const IRTYPE &, const SmallVectorImpl<const LoopInfo *> &lis);           \
  void add##NAME##Attr(IRTYPE &, const Loop &loop);                            \
  bool verify##NAME##Attr(const IRTYPE &, const Loop &loop,                    \
                          raw_ostream *os = nullptr);

#define DECL_ATTR_N(IRTYPE, NAME, IRNAME, ETY, ENAME, EN, NELEMS)              \
  std::optional<ETY> get##ENAME##From##NAME##Attr(const IRTYPE &);

#define DECL_ATTR_0(IRTYPE, NAME, IRNAME) void add##NAME##Attr(IRTYPE &);

#define DECL_ATTR_1(IRTYPE, NAME, IRNAME, TYPE)                                \
  void add##NAME##Attr(IRTYPE &, const TYPE &val);                             \
  std::optional<TYPE> get##NAME##Attr(const IRTYPE &f);                        \
  bool verify##NAME##Attr(const IRTYPE &, const TYPE &val,                     \
                          raw_ostream *os = nullptr);

#define DECL_ATTR_2(IRTYPE, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1,     \
                    EN1)                                                       \
  void add##NAME##Attr(IRTYPE &, const ETY0 &e0, const ETY1 &e1);              \
  bool verify##NAME##Attr(const IRTYPE &, const ETY0 &v0, const ETY1 &v1,      \
                          raw_ostream *os = nullptr);

#define DECL_ATTR_3(IRTYPE, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1,     \
                    EN1, ETY2, ENAME2, EN2)                                    \
  void add##NAME##Attr(IRTYPE &, const ETY0 &e0, const ETY1 &e1,               \
                       const ETY2 &e2);                                        \
  bool verify##NAME##Attr(const IRTYPE &, const ETY0 &v0, const ETY1 &v1,      \
                          const ETY2 &v2, raw_ostream *os = nullptr);

#define DECL_ATTR_4(IRTYPE, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1,     \
                    EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3)                 \
  void add##NAME##Attr(IRTYPE &, const ETY0 &e0, const ETY1 &e1,               \
                       const ETY2 &e2, const ETY3 &e3);                        \
  bool verify##NAME##Attr(const IRTYPE &, const ETY0 &v0, const ETY1 &v1,      \
                          const ETY2 &v2, const ETY3 &v3,                      \
                          raw_ostream *os = nullptr);

#define DECL_ATTR_5(IRTYPE, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1,     \
                    EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4,   \
                    EN4)                                                       \
  void add##NAME##Attr(IRTYPE &, const ETY0 &e0, const ETY1 &e1,               \
                       const ETY2 &e2, const ETY3 &e3, const ETY4 &e4);        \
  bool verify##NAME##Attr(const IRTYPE &, const ETY0 &v0, const ETY1 &v1,      \
                          const ETY2 &v2, const ETY3 &v3, const ETY4 &v4,      \
                          raw_ostream *os = nullptr);

#define DECL_ATTR_6(IRTYPE, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1,     \
                    EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4,   \
                    EN4, ETY5, ENAME5, EN5)                                    \
  void add##NAME##Attr(IRTYPE &, const ETY0 &e0, const ETY1 &e1,               \
                       const ETY2 &e2, const ETY3 &e3, const ETY4 &e4,         \
                       const ETY5 &e5);                                        \
  bool verify##NAME##Attr(const IRTYPE &, const ETY0 &v0, const ETY1 &v1,      \
                          const ETY2 &v2, const ETY3 &v3, const ETY4 &v4,      \
                          const ETY5 &v5, raw_ostream *os = nullptr);

#define DECL_ATTR_7(IRTYPE, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1,     \
                    EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4,   \
                    EN4, ETY5, ENAME5, EN5, ETY6, ENAME6, EN6)                 \
  void add##NAME##Attr(IRTYPE &, const ETY0 &e0, const ETY1 &e1,               \
                       const ETY2 &e2, const ETY3 &e3, const ETY4 &e4,         \
                       const ETY5 &e5, const ETY6 &e6);                        \
  bool verify##NAME##Attr(const IRTYPE &, const ETY0 &v0, const ETY1 &v1,      \
                          const ETY2 &v2, const ETY3 &v3, const ETY4 &v4,      \
                          const ETY5 &v5, const ETY6 &v6,                      \
                          raw_ostream *os = nullptr);

#define DECL_ATTR_8(IRTYPE, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1,     \
                    EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4,   \
                    EN4, ETY5, ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7,   \
                    EN7)                                                       \
  void add##NAME##Attr(IRTYPE &, const ETY0 &e0, const ETY1 &e1,               \
                       const ETY2 &e2, const ETY3 &e3, const ETY4 &e4,         \
                       const ETY5 &e5, const ETY6 &e6, const ETY7 &e7);        \
  bool verify##NAME##Attr(const IRTYPE &, const ETY0 &v0, const ETY1 &v1,      \
                          const ETY2 &v2, const ETY3 &v3, const ETY4 &v4,      \
                          const ETY5 &v5, const ETY6 &v6, const ETY7 &v7,      \
                          raw_ostream *os = nullptr);

#endif // KITSUNE_CORE_ATTRS_INTERNAL_H

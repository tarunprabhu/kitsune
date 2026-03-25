//==- AttrsCommon.h - Utilities common to Kitsune-specific attrs -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities shared by Kitsune-specific attributes.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_ATTRS_COMMON_H
#define KITSUNE_CORE_ATTRS_COMMON_H

#include "kitsune/Support/ToString.h"

namespace llvm {

#define ADD_0(ENUMCLASS, NAME, OBJ)                                            \
  do {                                                                         \
    ::addAttr((OBJ), ENUMCLASS::NAME, {});                                     \
  } while (0)

#define ADD_1(ENUMCLASS, NAME, OBJ, E0)                                        \
  do {                                                                         \
    LLVMContext &ctx = getContext(OBJ);                                        \
    Metadata *ops[] = {toMetadata(E0, ctx)};                                   \
    ::addAttr((OBJ), ENUMCLASS::NAME, ops);                                    \
  } while (0)

#define ADD_2(ENUMCLASS, NAME, OBJ, E0, E1)                                    \
  do {                                                                         \
    LLVMContext &ctx = getContext(OBJ);                                        \
    Metadata *ops[] = {toMetadata(E0, ctx), toMetadata(E1, ctx)};              \
    ::addAttr((OBJ), ENUMCLASS::NAME, ops);                                    \
  } while (0)

#define ADD_3(ENUMCLASS, NAME, OBJ, E0, E1, E2)                                \
  do {                                                                         \
    LLVMContext &ctx = getContext(OBJ);                                        \
    Metadata *ops[] = {toMetadata(E0, ctx), toMetadata(E1, ctx),               \
                       toMetadata(E2, ctx)};                                   \
    ::addAttr((OBJ), ENUMCLASS::NAME, ops);                                    \
  } while (0)

#define ADD_4(ENUMCLASS, NAME, OBJ, E0, E1, E2, E3)                            \
  do {                                                                         \
    LLVMContext &ctx = getContext(OBJ);                                        \
    Metadata *ops[] = {toMetadata(E0, ctx), toMetadata(E1, ctx),               \
                       toMetadata(E2, ctx), toMetadata(E3, ctx)};              \
    ::addAttr((OBJ), ENUMCLASS::NAME, ops);                                    \
  } while (0)

#define ADD_5(ENUMCLASS, NAME, OBJ, E0, E1, E2, E3, E4)                        \
  do {                                                                         \
    LLVMContext &ctx = getContext(OBJ);                                        \
    Metadata *ops[] = {toMetadata(E0, ctx), toMetadata(E1, ctx),               \
                       toMetadata(E2, ctx), toMetadata(E3, ctx),               \
                       toMetadata(E4, ctx)};                                   \
    ::addAttr((OBJ), ENUMCLASS::NAME, ops);                                    \
  } while (0)

#define ADD_6(ENUMCLASS, NAME, OBJ, E0, E1, E2, E3, E4, E5)                    \
  do {                                                                         \
    LLVMContext &ctx = getContext(OBJ);                                        \
    Metadata *ops[] = {toMetadata(E0, ctx), toMetadata(E1, ctx),               \
                       toMetadata(E2, ctx), toMetadata(E3, ctx),               \
                       toMetadata(E4, ctx), toMetadata(E5, ctx)};              \
    ::addAttr((OBJ), ENUMCLASS::NAME, ops);                                    \
  } while (0)

#define ADD_7(ENUMCLASS, NAME, OBJ, E0, E1, E2, E3, E4, E5, E6)                \
  do {                                                                         \
    LLVMContext &ctx = getContext(OBJ);                                        \
    Metadata *ops[] = {toMetadata(E0, ctx), toMetadata(E1, ctx),               \
                       toMetadata(E2, ctx), toMetadata(E3, ctx),               \
                       toMetadata(E4, ctx), toMetadata(E5, ctx),               \
                       toMetadata(E6, ctx)};                                   \
    ::addAttr((OBJ), ENUMCLASS::NAME, ops);                                    \
  } while (0)

#define ADD_8(ENUMCLASS, NAME, OBJ, E0, E1, E2, E3, E4, E5, E6, E7)            \
  do {                                                                         \
    LLVMContext &ctx = getContext(OBJ);                                        \
    Metadata *ops[] = {toMetadata(E0, ctx), toMetadata(E1, ctx),               \
                       toMetadata(E2, ctx), toMetadata(E3, ctx),               \
                       toMetadata(E4, ctx), toMetadata(E5, ctx),               \
                       toMetadata(E6, ctx), toMetadata(E7, ctx)};              \
    ::addAttr((OBJ), ENUMCLASS::NAME, ops);                                    \
  } while (0)

#define VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETYPE, ENAME)                       \
  do {                                                                         \
    if (has##NAME##Attr(OBJ)) {                                                \
      if (!get##ENAME##From##NAME##Attr(OBJ)) {                                \
        if (OS)                                                                \
          (*(OS)) << "Missing value of type '" << toString<ETYPE>()            \
                  << "' for element '" << #ENAME << "' in attribute '"         \
                  << IRNAME << "'\n";                                          \
        return false;                                                          \
      }                                                                        \
    }                                                                          \
    return true;                                                               \
  } while (0)

#define VERIFY_0(COND, IRNAME, OS)                                             \
  do {                                                                         \
    if (!(COND)) {                                                             \
      if (OS)                                                                  \
        (*OS) << "Unexpected value in attribute '" << IRNAME << "'\n";         \
      return false;                                                            \
    }                                                                          \
  } while (0)

#define VERIFY_1(OS, OBJ, NAME, IRNAME, TYPE)                                  \
  do {                                                                         \
    if (has##NAME##Attr(OBJ)) {                                                \
      if (!get##NAME##Attr(OBJ).has_value()) {                                 \
        if (OS)                                                                \
          (*(OS)) << "Missing value of type '" << toString<TYPE>()             \
                  << "' in attribute '" << IRNAME << "'\n";                    \
        return false;                                                          \
      }                                                                        \
    }                                                                          \
    return true;                                                               \
  } while (0)

#define VERIFY_2(OS, OBJ, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1)            \
  do {                                                                         \
    VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY0, ENAME0);                          \
    VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY1, ENAME1);                          \
  } while (0)

#define VERIFY_3(OS, OBJ, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2,      \
                 ENAME2)                                                       \
  do {                                                                         \
    VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY0, ENAME0);                          \
    VERIFY_2(OS, OBJ, NAME, IRNAME, ETY1, ENAME1, ETY2, ENAME2);               \
  } while (0)

#define VERIFY_4(OS, OBJ, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2,      \
                 ENAME2, ETY3, ENAME3)                                         \
  do {                                                                         \
    VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY0, ENAME0);                          \
    VERIFY_3(OS, OBJ, NAME, IRNAME, ETY1, ENAME1, ETY2, ENAME2, ETY3, ENAME3); \
  } while (0)

#define VERIFY_5(OS, OBJ, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2,      \
                 ENAME2, ETY3, ENAME3, ETY4, ENAME4)                           \
  do {                                                                         \
    VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY0, ENAME0);                          \
    VERIFY_4(OS, OBJ, NAME, IRNAME, ETY1, ENAME1, ETY2, ENAME2, ETY3, ENAME3,  \
             ETY4, ENAME4);                                                    \
  } while (0)

#define VERIFY_6(OS, OBJ, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2,      \
                 ENAME2, ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5)             \
  do {                                                                         \
    VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY0, ENAME0);                          \
    VERIFY_5(OS, OBJ, NAME, IRNAME, ETY1, ENAME1, ETY2, ENAME2, ETY3, ENAME3,  \
             ETY4, ENAME4, ETY5, ENAME5);                                      \
  } while (0)

#define VERIFY_7(OS, OBJ, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2,      \
                 ENAME2, ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5, ETY6,       \
                 ENAME6)                                                       \
  do {                                                                         \
    VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY0, ENAME0);                          \
    VERIFY_6(OS, OBJ, NAME, IRNAME, ETY1, ENAME1, ETY2, ENAME2, ETY3, ENAME3,  \
             ETY4, ENAME4, ETY5, ENAME5, ETY6, ENAME6);                        \
  } while (0)

#define VERIFY_8(OS, OBJ, NAME, IRNAME, ETY0, ENAME0, ETY1, ENAME1, ETY2,      \
                 ENAME2, ETY3, ENAME3, ETY4, ENAME4, ETY5, ENAME5, ETY6,       \
                 ENAME6, ETY7, ENAME7)                                         \
  do {                                                                         \
    VERIFY_IMPL(OS, OBJ, NAME, IRNAME, ETY0, ENAME0);                          \
    VERIFY_7(OS, OBJ, NAME, IRNAME, ETY1, ENAME1, ETY2, ENAME2, ETY3, ENAME3,  \
             ETY4, ENAME4, ETY5, ENAME5, ETY6, ENAME6, ETY7, ENAME7);          \
  } while (0)

} // namespace llvm

#endif // KITSUNE_CORE_ATTRS_COMMON_H

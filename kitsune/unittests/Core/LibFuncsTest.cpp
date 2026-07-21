//===- LibFuncTest.cpp - Unit tests for Kitsune's library functions -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/LibFuncs.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(LibFunc, getLibFuncName) {
#define GET_LIBFUNCS
#define LIBFUNC(NAME, LINKAGE_NAME, ...)                                       \
  EXPECT_EQ(getLibFuncName(KitFunc::NAME), LINKAGE_NAME);
#include "kitsune/Core/LibFuncs.inc"
}

#define NTYPES_(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14,   \
                _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26,    \
                _27, _28, _29, _30, _31, _32, _33, N, ...)                     \
  N
#define NTYPES(...)                                                            \
  NTYPES_(__VA_ARGS__, 33U, 32U, 31U, 30U, 29U, 28U, 27U, 26U, 25U, 24U, 23U,  \
          22U, 21U, 20U, 19U, 18U, 17U, 16U, 15U, 14U, 13U, 12U, 11U, 10U, 9U, \
          8U, 7U, 6U, 5U, 4U, 3U, 2U, 1U, X)

TEST(LibFunc, getLibFuncType) {
  LLVMContext ctx;

#define GET_LIBFUNCS
#define LIBFUNC(NAME, LINKAGE_NAME, ...)                                       \
  EXPECT_EQ(getLibFuncType(KitFunc::NAME, ctx)->getNumParams(),                \
            NTYPES(__VA_ARGS__) - 1);
#include "kitsune/Core/LibFuncs.inc"
}

TEST(libFunc, getOrInsertLibFunc) {
  LLVMContext ctx;
  Module m("", ctx);
  StringRef fname;
  FunctionCallee f, fre;

#define GET_LIBFUNCS
#define LIBFUNC(NAME, LINKAGE_NAME, ...)                                       \
  fname = getLibFuncName(KitFunc::NAME);                                       \
  EXPECT_FALSE(m.getFunction(fname));                                          \
  f = getOrInsertLibFunc(m, KitFunc::NAME);                                    \
  EXPECT_TRUE(m.getFunction(fname));                                           \
  fre = getOrInsertLibFunc(m, KitFunc::NAME);                                  \
  EXPECT_EQ(f.getCallee(), fre.getCallee());
#include "kitsune/Core/LibFuncs.inc"
}

} // namespace

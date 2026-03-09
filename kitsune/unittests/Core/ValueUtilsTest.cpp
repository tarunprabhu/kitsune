//===- ValueUtilsTest.cpp - Tests for Kitsune's value utilities -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ValueUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitValueUtils, isZero) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);
  Type *f32 = Type::getFloatTy(ctx);
  Type *f64 = Type::getDoubleTy(ctx);

  Constant *i0 = ConstantInt::get(i64, 0);
  Constant *f0 = ConstantFP::get(f32, 0);

  EXPECT_TRUE(isZero(i0));
  EXPECT_TRUE(isZero(i0, i64));
  EXPECT_TRUE(isZero(f0));
  EXPECT_TRUE(isZero(f0, f32));

  EXPECT_FALSE(isZero(i0, i32));
  EXPECT_FALSE(isZero(i0, f64));
}

TEST(KitValueUtils, isOne) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);
  Type *f32 = Type::getFloatTy(ctx);
  Type *f64 = Type::getDoubleTy(ctx);

  Constant *i1 = ConstantInt::get(i64, 1);
  Constant *f1 = ConstantFP::get(f32, 1);

  EXPECT_TRUE(isIntOne(i1));
  EXPECT_TRUE(isIntOne(i1, i64));

  EXPECT_FALSE(isIntOne(f1));
  EXPECT_FALSE(isIntOne(f1, f32));
  EXPECT_FALSE(isIntOne(i1, i32));
  EXPECT_FALSE(isIntOne(i1, f64));
}

} // namespace

//===- TypeUtilsTest.cpp - Unit tests for Kitsune's type utilities --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/TypeUtils.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"

#include "gtest/gtest.h"

using namespace llvm;

TEST(TypeUtils, isByteArrayTy) {
  LLVMContext ctx;
  Type *i8 = Type::getInt8Ty(ctx);          // i8
  Type *i16 = Type::getInt16Ty(ctx);        // i16
  Type *i64 = Type::getInt64Ty(ctx);        // i64
  Type *f32 = Type::getFloatTy(ctx);        // floats
  Type *arr_0 = ArrayType::get(i8, 0);      // [0 x i8]
  Type *arr_1 = ArrayType::get(i8, 1);      // [1 x i8]
  Type *arr_1_1 = ArrayType::get(arr_1, 1); // [1 x [1 x i8]]

  EXPECT_FALSE(isByteArrayTy(Type::getVoidTy(ctx)));        // void
  EXPECT_FALSE(isByteArrayTy(i64));                         // i64
  EXPECT_FALSE(isByteArrayTy(f32));                         // float
  EXPECT_FALSE(isByteArrayTy(PointerType::getUnqual(ctx))); // ptr
  EXPECT_FALSE(isByteArrayTy(ArrayType::get(i64, 0)));      // [0 x i64]
  EXPECT_FALSE(isByteArrayTy(ArrayType::get(i16, 1)));      // [1 x i16]
  EXPECT_FALSE(isByteArrayTy(StructType::create(arr_1)));   // { [1 x i8] }
  EXPECT_FALSE(isByteArrayTy(arr_1_1));                     // [1 x [1 x i8]]

  EXPECT_TRUE(isByteArrayTy(arr_0)); // [0 x i8]
  EXPECT_TRUE(isByteArrayTy(arr_1)); // [1 x i8]
}

TEST(TypeUtils, isZeroLenByteArrayTy) {
  LLVMContext ctx;
  Type *i8 = Type::getInt8Ty(ctx);          // i8
  Type *i16 = Type::getInt16Ty(ctx);        // i16
  Type *i64 = Type::getInt64Ty(ctx);        // i64
  Type *f32 = Type::getFloatTy(ctx);        // float
  Type *arr_0 = ArrayType::get(i8, 0);      // [0 x i8]
  Type *arr_1 = ArrayType::get(i8, 1);      // [1 x i8]
  Type *arr_0_0 = ArrayType::get(arr_0, 0); // [0 x [0 x i8]]

  EXPECT_FALSE(isZeroLenByteArrayTy(Type::getVoidTy(ctx)));        // void
  EXPECT_FALSE(isZeroLenByteArrayTy(i64));                         // i64
  EXPECT_FALSE(isZeroLenByteArrayTy(f32));                         // float
  EXPECT_FALSE(isZeroLenByteArrayTy(PointerType::getUnqual(ctx))); // ptr
  EXPECT_FALSE(isZeroLenByteArrayTy(ArrayType::get(i64, 0)));      // [0 x i64]
  EXPECT_FALSE(isZeroLenByteArrayTy(ArrayType::get(i16, 0)));      // [0 x i16]
  EXPECT_FALSE(isZeroLenByteArrayTy(StructType::create(arr_0))); // { [0 x i8] }
  EXPECT_FALSE(isZeroLenByteArrayTy(arr_0_0)); // [0 x [0 x i8]]
  EXPECT_FALSE(isZeroLenByteArrayTy(arr_1));   // [1 x i8]

  EXPECT_TRUE(isZeroLenByteArrayTy(arr_0)); // [0 x i8]
}

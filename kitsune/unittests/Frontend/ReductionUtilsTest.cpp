//===- ReductionUtilsTest.cpp - Unit tests for reduction utilities --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Frontend/ReductionUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitReductionUtils, fromInt) {
  EXPECT_EQ(fromInt<ReduceOp>(0), ReduceOp::Custom);
  EXPECT_EQ(fromInt<ReduceOp>(1), ReduceOp::BAnd);
  EXPECT_EQ(fromInt<ReduceOp>(2), ReduceOp::BOr);
  EXPECT_EQ(fromInt<ReduceOp>(3), ReduceOp::BXor);
  EXPECT_EQ(fromInt<ReduceOp>(4), ReduceOp::LAnd);
  EXPECT_EQ(fromInt<ReduceOp>(5), ReduceOp::LOr);
  EXPECT_EQ(fromInt<ReduceOp>(6), ReduceOp::LXor);
  EXPECT_EQ(fromInt<ReduceOp>(7), ReduceOp::Max);
  EXPECT_EQ(fromInt<ReduceOp>(8), ReduceOp::MaxLoc);
  EXPECT_EQ(fromInt<ReduceOp>(9), ReduceOp::Min);
  EXPECT_EQ(fromInt<ReduceOp>(10), ReduceOp::MinLoc);
  EXPECT_EQ(fromInt<ReduceOp>(11), ReduceOp::Prod);
  EXPECT_EQ(fromInt<ReduceOp>(12), ReduceOp::Sum);

  EXPECT_FALSE(fromInt<ReduceOp>(-1).has_value());
  EXPECT_FALSE(fromInt<ReduceOp>(13).has_value());
}

TEST(KitReductionUtils, getUnitBAnd) {
  auto check = [](Type *ty) {
    Constant *c = getUnitValueFor(ReduceOp::BAnd, ty);

    EXPECT_EQ(c->getType(), ty);
    EXPECT_TRUE(isa<ConstantInt>(c));
    EXPECT_TRUE(cast<ConstantInt>(c)->isMaxValue(/*isSigned=*/false));
  };

  LLVMContext ctx;
  Type *i1 = Type::getInt1Ty(ctx);
  Type *i8 = Type::getInt8Ty(ctx);
  Type *i16 = Type::getInt16Ty(ctx);
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);

  check(i1);
  check(i8);
  check(i16);
  check(i32);
  check(i64);
}

TEST(KitReductionUtils, getUnitBOr) {
  auto check = [](Type *ty) {
    Constant *c = getUnitValueFor(ReduceOp::BOr, ty);

    EXPECT_EQ(c->getType(), ty);
    EXPECT_TRUE(isa<ConstantInt>(c));
    EXPECT_TRUE(cast<ConstantInt>(c)->isZero());
  };

  LLVMContext ctx;
  Type *i1 = Type::getInt1Ty(ctx);
  Type *i8 = Type::getInt8Ty(ctx);
  Type *i16 = Type::getInt16Ty(ctx);
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);

  check(i1);
  check(i8);
  check(i16);
  check(i32);
  check(i64);
}

TEST(KitReductionUtils, getUnitBXor) {
  auto check = [](Type *ty) {
    Constant *c = getUnitValueFor(ReduceOp::BXor, ty);

    EXPECT_EQ(c->getType(), ty);
    EXPECT_TRUE(isa<ConstantInt>(c));
    EXPECT_TRUE(cast<ConstantInt>(c)->isZero());
  };

  LLVMContext ctx;
  Type *i1 = Type::getInt1Ty(ctx);
  Type *i8 = Type::getInt8Ty(ctx);
  Type *i16 = Type::getInt16Ty(ctx);
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);

  check(i1);
  check(i8);
  check(i16);
  check(i32);
  check(i64);
}

TEST(KitReductionUtils, getUnitLAnd) {
  auto check = [](Type *ty) {
    Constant *c = getUnitValueFor(ReduceOp::LAnd, ty);
    EXPECT_EQ(c->getType(), ty);
    EXPECT_TRUE(isa<ConstantInt>(c));
    EXPECT_TRUE(cast<ConstantInt>(c)->isOne());
  };

  // In principle, this should be tested with any integer type, but the frontend
  // will restrict the operands to LAnd to be booleans which are typically
  // represented as i8.
  LLVMContext ctx;
  Type *i1 = Type::getInt1Ty(ctx);
  Type *i8 = Type::getInt8Ty(ctx);

  check(i1);
  check(i8);
}

TEST(KitReductionUtils, getUnitLOr) {
  auto check = [](Type *ty) {
    Constant *c = getUnitValueFor(ReduceOp::LOr, ty);
    EXPECT_EQ(c->getType(), ty);
    EXPECT_TRUE(isa<ConstantInt>(c));
    EXPECT_TRUE(cast<ConstantInt>(c)->isZero());
  };

  // In principle, this should be tested with any integer type, but the frontend
  // will restrict the operands to LAnd to be booleans which are typically
  // represented as i8.
  LLVMContext ctx;
  Type *i1 = Type::getInt1Ty(ctx);
  Type *i8 = Type::getInt8Ty(ctx);

  check(i1);
  check(i8);
}

TEST(KitReductionUtils, getUnitLXor) {
  auto check = [](Type *ty) {
    Constant *c = getUnitValueFor(ReduceOp::LXor, ty);
    EXPECT_EQ(c->getType(), ty);
    EXPECT_TRUE(isa<ConstantInt>(c));
    EXPECT_TRUE(cast<ConstantInt>(c)->isZero());
  };

  // In principle, this should be tested with any integer type, but the frontend
  // will restrict the operands to LAnd to be booleans which are typically
  // represented as i8.
  LLVMContext ctx;
  Type *i1 = Type::getInt1Ty(ctx);
  Type *i8 = Type::getInt8Ty(ctx);

  check(i1);
  check(i8);
}

TEST(KitReductionUtils, getUnitMax) {
  auto checkInt = [](Type *ty, bool isSigned) {
    Constant *c = getUnitValueFor(ReduceOp::Max, ty, isSigned);

    EXPECT_EQ(c->getType(), ty);
    EXPECT_TRUE(isa<ConstantInt>(c));
    EXPECT_TRUE(cast<ConstantInt>(c)->isMinValue(isSigned));
  };

  auto checkInts = [&](Type *ty) {
    checkInt(ty, /*isSigned=*/false);
    checkInt(ty, /*isSigned=*/true);
  };

  auto checkFP = [](Type *ty, APFloat expected) {
    Constant *c = getUnitValueFor(ReduceOp::Max, ty);

    EXPECT_EQ(c->getType(), ty);
    EXPECT_TRUE(isa<ConstantFP>(c));
    EXPECT_EQ(cast<ConstantFP>(c)->getValue().compare(expected),
              APFloat::cmpEqual);
  };

  LLVMContext ctx;
  Type *i8 = Type::getInt8Ty(ctx);
  Type *i16 = Type::getInt16Ty(ctx);
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);
  Type *f32 = Type::getFloatTy(ctx);
  Type *f64 = Type::getDoubleTy(ctx);

  checkInts(i8);
  checkInts(i16);
  checkInts(i32);
  checkInts(i64);
  checkFP(f32, APFloat(std::numeric_limits<float>::min()));
  checkFP(f64, APFloat(std::numeric_limits<double>::min()));
}

TEST(KitReductionUtils, getUnitMaxLoc) {
  auto checkInt = [](Type *ty, bool isSigned) {
    Constant *c = getUnitValueFor(ReduceOp::MaxLoc, ty, isSigned);

    EXPECT_EQ(c->getType(), ty);
    EXPECT_TRUE(isa<ConstantInt>(c));
    EXPECT_TRUE(cast<ConstantInt>(c)->isMinValue(isSigned));
  };

  auto checkInts = [&](Type *ty) {
    checkInt(ty, /*isSigned=*/false);
    checkInt(ty, /*isSigned=*/true);
  };

  auto checkFP = [](Type *ty, APFloat expected) {
    Constant *c = getUnitValueFor(ReduceOp::MaxLoc, ty);

    EXPECT_EQ(c->getType(), ty);
    EXPECT_TRUE(isa<ConstantFP>(c));
    EXPECT_EQ(cast<ConstantFP>(c)->getValue().compare(expected),
              APFloat::cmpEqual);
  };

  LLVMContext ctx;
  Type *i8 = Type::getInt8Ty(ctx);
  Type *i16 = Type::getInt16Ty(ctx);
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);
  Type *f32 = Type::getFloatTy(ctx);
  Type *f64 = Type::getDoubleTy(ctx);

  checkInts(i8);
  checkInts(i16);
  checkInts(i32);
  checkInts(i64);
  checkFP(f32, APFloat(std::numeric_limits<float>::min()));
  checkFP(f64, APFloat(std::numeric_limits<double>::min()));
}

TEST(KitReductionUtils, getUnitMin) {
  auto checkInt = [](Type *ty, bool isSigned) {
    Constant *c = getUnitValueFor(ReduceOp::Min, ty, isSigned);

    EXPECT_EQ(c->getType(), ty);
    EXPECT_TRUE(isa<ConstantInt>(c));
    EXPECT_TRUE(cast<ConstantInt>(c)->isMaxValue(isSigned));
  };

  auto checkInts = [&](Type *ty) {
    checkInt(ty, /*isSigned=*/false);
    checkInt(ty, /*isSigned=*/true);
  };

  auto checkFP = [](Type *ty, APFloat expected) {
    Constant *c = getUnitValueFor(ReduceOp::Min, ty);

    EXPECT_EQ(c->getType(), ty);
    EXPECT_TRUE(isa<ConstantFP>(c));
    EXPECT_EQ(cast<ConstantFP>(c)->getValue().compare(expected),
              APFloat::cmpEqual);
  };

  LLVMContext ctx;
  Type *i8 = Type::getInt8Ty(ctx);
  Type *i16 = Type::getInt16Ty(ctx);
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);
  Type *f32 = Type::getFloatTy(ctx);
  Type *f64 = Type::getDoubleTy(ctx);

  checkInts(i8);
  checkInts(i16);
  checkInts(i32);
  checkInts(i64);
  checkFP(f32, APFloat(std::numeric_limits<float>::max()));
  checkFP(f64, APFloat(std::numeric_limits<double>::max()));
}

TEST(KitReductionUtils, getUnitMinLoc) {
  auto checkInt = [](Type *ty, bool isSigned) {
    Constant *c = getUnitValueFor(ReduceOp::MinLoc, ty, isSigned);

    EXPECT_EQ(c->getType(), ty);
    EXPECT_TRUE(isa<ConstantInt>(c));
    EXPECT_TRUE(cast<ConstantInt>(c)->isMaxValue(isSigned));
  };

  auto checkInts = [&](Type *ty) {
    checkInt(ty, /*isSigned=*/false);
    checkInt(ty, /*isSigned=*/true);
  };

  auto checkFP = [](Type *ty, APFloat expected) {
    Constant *c = getUnitValueFor(ReduceOp::MinLoc, ty);

    EXPECT_EQ(c->getType(), ty);
    EXPECT_TRUE(isa<ConstantFP>(c));
    EXPECT_EQ(cast<ConstantFP>(c)->getValue().compare(expected),
              APFloat::cmpEqual);
  };

  LLVMContext ctx;
  Type *i8 = Type::getInt8Ty(ctx);
  Type *i16 = Type::getInt16Ty(ctx);
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);
  Type *f32 = Type::getFloatTy(ctx);
  Type *f64 = Type::getDoubleTy(ctx);

  checkInts(i8);
  checkInts(i16);
  checkInts(i32);
  checkInts(i64);
  checkFP(f32, APFloat(std::numeric_limits<float>::max()));
  checkFP(f64, APFloat(std::numeric_limits<double>::max()));
}

TEST(KitReductionUtils, getUnitProd) {
  auto checkInt = [](Type *ty, bool isSigned) {
    Constant *c = getUnitValueFor(ReduceOp::Prod, ty, isSigned);

    EXPECT_EQ(c->getType(), ty);
    EXPECT_TRUE(isa<ConstantInt>(c));
    EXPECT_TRUE(cast<ConstantInt>(c)->isOne());
  };

  auto checkInts = [&](Type *ty) {
    checkInt(ty, /*isSigned=*/false);
    checkInt(ty, /*isSigned=*/true);
  };

  auto checkFP = [](Type *ty) {
    Constant *c = getUnitValueFor(ReduceOp::Prod, ty);

    EXPECT_EQ(c->getType(), ty);
    EXPECT_TRUE(isa<ConstantFP>(c));

    const APFloat &v = cast<ConstantFP>(c)->getValue();
    APFloat one = APFloat::getOne(v.getSemantics(), /*Negative=*/false);
    EXPECT_EQ(v.compare(one), APFloat::cmpEqual);
  };

  LLVMContext ctx;
  Type *i8 = Type::getInt8Ty(ctx);
  Type *i16 = Type::getInt16Ty(ctx);
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);
  Type *f32 = Type::getFloatTy(ctx);
  Type *f64 = Type::getDoubleTy(ctx);

  checkInts(i8);
  checkInts(i16);
  checkInts(i32);
  checkInts(i64);
  checkFP(f32);
  checkFP(f64);
}

TEST(KitReductionUtils, getUnitSum) {
  auto checkInt = [](Type *ty, bool isSigned) {
    Constant *c = getUnitValueFor(ReduceOp::Sum, ty, isSigned);

    EXPECT_EQ(c->getType(), ty);
    EXPECT_TRUE(isa<ConstantInt>(c));
    EXPECT_TRUE(cast<ConstantInt>(c)->isZero());
  };

  auto checkInts = [&](Type *ty) {
    checkInt(ty, /*isSigned=*/false);
    checkInt(ty, /*isSigned=*/true);
  };

  auto checkFP = [](Type *ty) {
    Constant *c = getUnitValueFor(ReduceOp::Sum, ty);

    EXPECT_EQ(c->getType(), ty);
    EXPECT_TRUE(isa<ConstantFP>(c));
    EXPECT_TRUE(cast<ConstantFP>(c)->isZero());
  };

  LLVMContext ctx;
  Type *i8 = Type::getInt8Ty(ctx);
  Type *i16 = Type::getInt16Ty(ctx);
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);
  Type *f32 = Type::getFloatTy(ctx);
  Type *f64 = Type::getDoubleTy(ctx);

  checkInts(i8);
  checkInts(i16);
  checkInts(i32);
  checkInts(i64);
  checkFP(f32);
  checkFP(f64);
}

} // namespace

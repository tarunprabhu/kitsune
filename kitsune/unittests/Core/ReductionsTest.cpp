//===- ReductionsTest.cpp - Unit tests for reduction utilities ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/Reductions.h"
#include "kitsune/Core/ModuleUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitReductions, fromInt) {
  EXPECT_EQ(fromInt<ReduceOp>(0), ReduceOp::Custom);
  EXPECT_EQ(fromInt<ReduceOp>(1), ReduceOp::And);
  EXPECT_EQ(fromInt<ReduceOp>(2), ReduceOp::Or);
  EXPECT_EQ(fromInt<ReduceOp>(3), ReduceOp::Xor);
  EXPECT_EQ(fromInt<ReduceOp>(5), ReduceOp::Add);
  EXPECT_EQ(fromInt<ReduceOp>(6), ReduceOp::FAdd);
  EXPECT_EQ(fromInt<ReduceOp>(7), ReduceOp::Mul);
  EXPECT_EQ(fromInt<ReduceOp>(8), ReduceOp::FMul);
  EXPECT_EQ(fromInt<ReduceOp>(16), ReduceOp::FMax);
  EXPECT_EQ(fromInt<ReduceOp>(17), ReduceOp::FMaximum);
  EXPECT_EQ(fromInt<ReduceOp>(18), ReduceOp::FMaximumNum);
  EXPECT_EQ(fromInt<ReduceOp>(20), ReduceOp::FMin);
  EXPECT_EQ(fromInt<ReduceOp>(21), ReduceOp::FMinimum);
  EXPECT_EQ(fromInt<ReduceOp>(22), ReduceOp::FMinimumNum);
  EXPECT_EQ(fromInt<ReduceOp>(24), ReduceOp::SMax);
  EXPECT_EQ(fromInt<ReduceOp>(25), ReduceOp::SMin);
  EXPECT_EQ(fromInt<ReduceOp>(26), ReduceOp::UMax);
  EXPECT_EQ(fromInt<ReduceOp>(27), ReduceOp::UMin);

  EXPECT_FALSE(fromInt<ReduceOp>(-1).has_value());
  EXPECT_FALSE(fromInt<ReduceOp>(32).has_value());
}

TEST(KitReductions, getUnitAnd) {
  auto check = [](Type *ty) {
    Constant *c = getUnitValue(ReduceOp::And, ty);

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

TEST(KitReductions, getUnitOr) {
  auto check = [](Type *ty) {
    Constant *c = getUnitValue(ReduceOp::Or, ty);

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

TEST(KitReductions, getUnitXor) {
  auto check = [](Type *ty) {
    Constant *c = getUnitValue(ReduceOp::Xor, ty);

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

TEST(KitReductions, getUnitMax) {
  auto checkInt = [](Constant *actual, Type *ty, bool isSigned) {
    EXPECT_EQ(actual->getType(), ty);
    EXPECT_TRUE(isa<ConstantInt>(actual));
    EXPECT_TRUE(cast<ConstantInt>(actual)->isMinValue(isSigned));
  };

  auto checkInts = [&checkInt](Type *ty) {
    for (ReduceOp op : {ReduceOp::FMax, ReduceOp::FMaximum,
                        ReduceOp::FMaximumNum, ReduceOp::SMax})
      checkInt(getUnitValue(op, ty), ty, /*isSigned=*/true);
    checkInt(getUnitValue(ReduceOp::UMax, ty), ty, /*isSigned=*/false);
  };

  auto checkFP = [](Constant *actual, Type *ty, APFloat expected) {
    EXPECT_EQ(actual->getType(), ty);
    EXPECT_TRUE(isa<ConstantFP>(actual));
    EXPECT_EQ(cast<ConstantFP>(actual)->getValue().compare(expected),
              APFloat::cmpEqual);
  };

  auto checkFPs = [&checkFP](Type *ty, APFloat expected) {
    for (ReduceOp op :
         {ReduceOp::FMax, ReduceOp::FMaximum, ReduceOp::FMaximumNum})
      checkFP(getUnitValue(op, ty), ty, expected);
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
  checkFPs(f32, APFloat(std::numeric_limits<float>::min()));
  checkFPs(f64, APFloat(std::numeric_limits<double>::min()));
}

TEST(KitReductions, getUnitMin) {
  auto checkInt = [](Constant *actual, Type *ty, bool isSigned) {
    EXPECT_EQ(actual->getType(), ty);
    EXPECT_TRUE(isa<ConstantInt>(actual));
    EXPECT_TRUE(cast<ConstantInt>(actual)->isMaxValue(isSigned));
  };

  auto checkInts = [&checkInt](Type *ty) {
    for (ReduceOp op : {ReduceOp::FMin, ReduceOp::FMinimum,
                        ReduceOp::FMinimumNum, ReduceOp::SMin})
      checkInt(getUnitValue(op, ty), ty, /*isSigned=*/true);
    checkInt(getUnitValue(ReduceOp::UMin, ty), ty, /*isSigned=*/false);
  };

  auto checkFP = [](Constant *actual, Type *ty, APFloat expected) {
    EXPECT_EQ(actual->getType(), ty);
    EXPECT_TRUE(isa<ConstantFP>(actual));
    EXPECT_EQ(cast<ConstantFP>(actual)->getValue().compare(expected),
              APFloat::cmpEqual);
  };

  auto checkFPs = [&checkFP](Type *ty, APFloat expected) {
    for (ReduceOp op :
         {ReduceOp::FMin, ReduceOp::FMinimum, ReduceOp::FMinimumNum})
      checkFP(getUnitValue(op, ty), ty, expected);
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
  checkFPs(f32, APFloat(std::numeric_limits<float>::max()));
  checkFPs(f64, APFloat(std::numeric_limits<double>::max()));
}

TEST(KitReductions, getUnitMul) {
  auto checkInt = [](Type *ty, bool isSigned) {
    Constant *c = getUnitValue(ReduceOp::Mul, ty);

    EXPECT_EQ(c->getType(), ty);
    EXPECT_TRUE(isa<ConstantInt>(c));
    EXPECT_TRUE(cast<ConstantInt>(c)->isOne());
  };

  auto checkInts = [&](Type *ty) {
    checkInt(ty, /*isSigned=*/false);
    checkInt(ty, /*isSigned=*/true);
  };

  auto checkFP = [](Type *ty) {
    Constant *c = getUnitValue(ReduceOp::FMul, ty);

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

TEST(KitReductions, getUnitAdd) {
  auto checkInt = [](Type *ty, bool isSigned) {
    Constant *c = getUnitValue(ReduceOp::Add, ty);

    EXPECT_EQ(c->getType(), ty);
    EXPECT_TRUE(isa<ConstantInt>(c));
    EXPECT_TRUE(cast<ConstantInt>(c)->isZero());
  };

  auto checkInts = [&](Type *ty) {
    checkInt(ty, /*isSigned=*/false);
    checkInt(ty, /*isSigned=*/true);
  };

  auto checkFP = [](Type *ty) {
    Constant *c = getUnitValue(ReduceOp::FAdd, ty);

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

TEST(KitReductions, getReducerType) {
  LLVMContext ctx;
  Module m("", ctx);

  PointerType *ptr = PointerType::getUnqual(ctx);
  Type *i32 = Type::getInt32Ty(ctx);

  Constant *cnull = ConstantPointerNull::get(ptr);
  Constant *zero = ConstantInt::get(i32, 0);
  Constant *one = ConstantInt::get(i32, 1);
  Constant *four = ConstantInt::get(i32, 4);
  Constant *five = ConstantInt::get(i32, 5);
  Constant *ten = ConstantInt::get(i32, 10);

  Function *f = getOrInsertFunction(m, "test", i32);
  BasicBlock *bb = BasicBlock::Create(ctx, "", f);
  IRBuilder<> builder(bb);

  CallInst *cf =
      builder.CreateIntrinsic(Intrinsic::kit_reduce_0, {i32, i32},
                              {one, five, cnull, four, ten, zero, cnull});
  ReductionInfo redf(cf);
  FunctionType *tyf = redf.getReducerType();

  EXPECT_EQ(tyf->getNumParams(), 2U);
  EXPECT_EQ(tyf->getParamType(0), ptr);
  EXPECT_EQ(tyf->getParamType(1), i32);
  EXPECT_FALSE(tyf->isVarArg());
  EXPECT_EQ(redf.getReducerArgs(), (SmallVector<Value *, 2>{cnull, ten}));

  CallInst *cv = builder.CreateIntrinsic(
      Intrinsic::kit_reduce_0, {i32, i32, ptr},
      {one, five, cnull, four, ten, zero, cnull, cnull});
  ReductionInfo redv(cv);
  FunctionType *tyv = redv.getReducerType();

  EXPECT_FALSE(tyv->isVarArg());
  EXPECT_EQ(redv.getReducerArgs(),
            (SmallVector<Value *, 2>{cnull, ten, cnull}));
}

} // namespace

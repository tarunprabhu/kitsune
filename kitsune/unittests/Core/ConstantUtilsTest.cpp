//===- ConstantUtilsTest.cpp - Unit tests for constant utilities ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ConstantUtils.h"
#include "TestUtils.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitConstantUtils, createConstString) {
  LLVMContext ctx;
  StringRef s1 = "test1";
  StringRef s2 = "test2";
  std::unique_ptr<Module> m = parseIR(ctx, R"(
    @farr = constant [2 x i32] [i32 11, i32 22]
  )");

  GlobalVariable *g1 = createConstString(s1, *m);

  EXPECT_FALSE(g1->hasName());
  EXPECT_TRUE(g1->hasInitializer());
  EXPECT_TRUE(isa<ConstantDataArray>(g1->getInitializer()));
  EXPECT_EQ(cast<ConstantDataArray>(g1->getInitializer())->getAsCString(), s1);
  EXPECT_EQ(m->global_size(), 2U);

  // It doesn't matter what other parameters are passed to the function, if a
  // global string with the given initializer already exists, it will be
  // returned.
  EXPECT_EQ(createConstString(s1, *m), g1);
  EXPECT_EQ(createConstString(s1, *m, "eccentrica"), g1);
  EXPECT_FALSE(createConstString(s1, *m, "eccentrica")->hasName());

  // When asking for a different string, make sure that works too.
  GlobalVariable *g2 = createConstString(s2, *m, "s2");

  EXPECT_EQ(m->global_size(), 3U);
  EXPECT_EQ(g2->getName(), "s2");
}

TEST(KitConstantUtils, stripCasts) {
  LLVMContext ctx;
  Type *i64 = Type::getInt64Ty(ctx);
  Type *f32 = Type::getFloatTy(ctx);
  PointerType *ptr = PointerType::getUnqual(ctx);
  PointerType *ptr11 = PointerType::get(ctx, 11);

  Constant *cnull = ConstantPointerNull::get(ptr);
  Constant *cint = ConstantInt::get(i64, 79);
  Constant *cfp = ConstantFP::get(f32, 3.14);
  Constant *cptr = ConstantExpr::getIntToPtr(cint, ptr);
  Constant *cptr11 = ConstantExpr::getAddrSpaceCast(cptr, ptr11);

  EXPECT_EQ(stripCasts((const Constant *)nullptr), nullptr);

  EXPECT_EQ(stripCasts(cint), cint);
  EXPECT_EQ(stripCasts(cnull), cnull);
  EXPECT_EQ(stripCasts(cfp), cfp);
  EXPECT_EQ(stripCasts(cptr), cint);
  EXPECT_EQ(stripCasts(cptr11), cint);

  EXPECT_EQ(stripCasts((const Constant *)cint), cint);
  EXPECT_EQ(stripCasts((const Constant *)cnull), cnull);
  EXPECT_EQ(stripCasts((const Constant *)cfp), cfp);
  EXPECT_EQ(stripCasts((const Constant *)cptr), cint);
  EXPECT_EQ(stripCasts((const Constant *)cptr11), cint);
}

TEST(KitConstantUtils, fromConstant) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);
  Type *f32 = Type::getFloatTy(ctx);
  Type *f64 = Type::getDoubleTy(ctx);
  PointerType *ptr = PointerType::getUnqual(ctx);

  Constant *cnull = ConstantPointerNull::get(ptr);
  Constant *ci32 = ConstantInt::get(i32, 99);
  Constant *ci64 = ConstantInt::get(i64, 79);
  Constant *cf32 = ConstantFP::get(f32, 3.14F);
  Constant *cf64 = ConstantFP::get(f64, 2.71828);
  Constant *cstr = ConstantDataArray::getString(ctx, "glasgow");
  Constant *one = ConstantInt::get(i32, 1);

  EXPECT_FALSE(fromConstant<int32_t>(*ci64));
  EXPECT_FALSE(fromConstant<int32_t>(*cf32));
  EXPECT_FALSE(fromConstant<int64_t>(*ci32));
  EXPECT_FALSE(fromConstant<int64_t>(*cf64));

  EXPECT_FALSE(fromConstant<float>(*cf64));
  EXPECT_FALSE(fromConstant<float>(*ci32));
  EXPECT_FALSE(fromConstant<double>(*cf32));
  EXPECT_FALSE(fromConstant<double>(*ci64));

  EXPECT_FALSE(fromConstant<int32_t>(*cstr));
  EXPECT_FALSE(fromConstant<int64_t>(*cnull));
  EXPECT_FALSE(fromConstant<double>(*cnull));
  EXPECT_FALSE(fromConstant<StringRef>(*cnull));

  EXPECT_FALSE(fromConstant<TTID>(*ci64));
  EXPECT_FALSE(fromConstant<TTID>(*cf64));
  EXPECT_FALSE(fromConstant<TTID>(*cf32));
  EXPECT_EQ(fromConstant<TTID>(*one), TTID::Serial);
}

TEST(KitConstantUtils, toConstant) {
  std::string s = "soas";
  StringLiteral lit = "lse";

  LLVMContext ctx;
  Type *i1 = Type::getInt1Ty(ctx);
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);
  Type *f32 = Type::getFloatTy(ctx);
  Type *f64 = Type::getDoubleTy(ctx);

  Constant *cTrue = toConstant(true, ctx);
  Constant *cFalse = toConstant(false, ctx);
  Constant *ci32 = toConstant(1, ctx);
  Constant *ci64 = toConstant(1L, ctx);
  Constant *cf32 = toConstant(1.0F, ctx);
  Constant *cf64 = toConstant(1.0, ctx);
  Constant *cTTID = toConstant(TTID::Serial, ctx);
  Constant *cStrat = toConstant(TapirSpawnStrategy::Sequential, ctx);
  Constant *ccstr = toConstant("edinburgh", ctx);
  Constant *cstring = toConstant(s, ctx);
  Constant *clit = toConstant(lit, ctx);

  EXPECT_EQ(cTrue->getType(), i1);
  EXPECT_EQ(cFalse->getType(), i1);
  EXPECT_EQ(ci32->getType(), i32);
  EXPECT_EQ(ci64->getType(), i64);
  EXPECT_EQ(cf32->getType(), f32);
  EXPECT_EQ(cf64->getType(), f64);

  EXPECT_EQ(cTTID->getType(), i32);
  EXPECT_EQ(cStrat->getType(), i32);

  EXPECT_EQ(fromConstant<bool>(*cTrue), true);
  EXPECT_EQ(fromConstant<bool>(*cFalse), false);
  EXPECT_EQ(fromConstant<int32_t>(*ci32), 1);
  EXPECT_EQ(fromConstant<int64_t>(*ci64), 1L);
  EXPECT_EQ(fromConstant<float>(*cf32), 1.0F);
  EXPECT_EQ(fromConstant<double>(*cf64), 1.0);
  EXPECT_EQ(fromConstant<TTID>(*cTTID), TTID::Serial);
  EXPECT_EQ(fromConstant<TapirSpawnStrategy>(*cStrat),
            TapirSpawnStrategy::Sequential);
  EXPECT_EQ(fromConstant<StringRef>(*ccstr), "edinburgh");
  EXPECT_EQ(fromConstant<StringRef>(*cstring), "soas");
  EXPECT_EQ(fromConstant<StringRef>(*clit), "lse");
}

TEST(KitConstantUtils, getZero) {
  LLVMContext ctx;
  Constant *i1 = getZero(Type::getInt1Ty(ctx));
  Constant *i8 = getZero(Type::getInt8Ty(ctx));
  Constant *i16 = getZero(Type::getInt16Ty(ctx));
  Constant *i32 = getZero(Type::getInt32Ty(ctx));
  Constant *i64 = getZero(Type::getInt64Ty(ctx));
  Constant *f32 = getZero(Type::getFloatTy(ctx));
  Constant *f64 = getZero(Type::getDoubleTy(ctx));

  EXPECT_TRUE(i1->isNullValue());
  EXPECT_TRUE(i8->isNullValue());
  EXPECT_TRUE(i16->isNullValue());
  EXPECT_TRUE(i32->isNullValue());
  EXPECT_TRUE(i64->isNullValue());
  EXPECT_TRUE(f32->isNullValue());
  EXPECT_TRUE(f64->isNullValue());

  EXPECT_TRUE(i1->getType()->isIntegerTy(1));
  EXPECT_TRUE(i8->getType()->isIntegerTy(8));
  EXPECT_TRUE(i16->getType()->isIntegerTy(16));
  EXPECT_TRUE(i32->getType()->isIntegerTy(32));
  EXPECT_TRUE(i64->getType()->isIntegerTy(64));
  EXPECT_TRUE(f32->getType()->isFloatTy());
  EXPECT_TRUE(f64->getType()->isDoubleTy());
}

TEST(KitConstantUtils, getOne) {
  LLVMContext ctx;
  Constant *i1 = getOne(Type::getInt1Ty(ctx));
  Constant *i8 = getOne(Type::getInt8Ty(ctx));
  Constant *i16 = getOne(Type::getInt16Ty(ctx));
  Constant *i32 = getOne(Type::getInt32Ty(ctx));
  Constant *i64 = getOne(Type::getInt64Ty(ctx));
  Constant *f32 = getOne(Type::getFloatTy(ctx));
  Constant *f64 = getOne(Type::getDoubleTy(ctx));

  EXPECT_TRUE(i1->isOneValue());
  EXPECT_TRUE(i8->isOneValue());
  EXPECT_TRUE(i16->isOneValue());
  EXPECT_TRUE(i32->isOneValue());
  EXPECT_TRUE(i64->isOneValue());
  EXPECT_EQ(cast<ConstantFP>(f32)->getValue(), APFloat(1.0f));
  EXPECT_EQ(cast<ConstantFP>(f64)->getValue(), APFloat(1.0));

  EXPECT_TRUE(i1->getType()->isIntegerTy(1));
  EXPECT_TRUE(i8->getType()->isIntegerTy(8));
  EXPECT_TRUE(i16->getType()->isIntegerTy(16));
  EXPECT_TRUE(i32->getType()->isIntegerTy(32));
  EXPECT_TRUE(i64->getType()->isIntegerTy(64));
  EXPECT_TRUE(f32->getType()->isFloatTy());
  EXPECT_TRUE(f64->getType()->isDoubleTy());
}

struct Interop {
  int i;
  char s[5];
};

TEST(KitConstantUtils, getInteroperableConst) {
  auto toString = [](const Constant &c) -> std::string {
    std::string buf;
    raw_string_ostream os(buf);

    os << c;
    os.flush();

    return buf;
  };

  LLVMContext ctx;
  Interop o;

  // Set all bytes of the struct to some silly marker value that we can later
  // recognize. Only the first 9 bytes of this contain actual data, so the
  // sentinels should be left untouched.
  memset(&o, 0xbe, sizeof(Interop));

  // Now add proper data.
  o.i = 11;
  memcpy(o.s, "cafe", 5);

  Constant *c = toConstant(o, ctx);
  std::string cstr = toString(*c);

  EXPECT_TRUE(isa<ConstantDataArray>(c));
  EXPECT_EQ(cstr, "[12 x i8] c\"\\0B\\00\\00\\00cafe\\00\\BE\\BE\\BE\"");
}

} // namespace

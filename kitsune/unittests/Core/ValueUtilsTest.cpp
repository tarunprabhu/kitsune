//===- ValueUtilsTest.cpp - Tests for Kitsune's value utilities -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ValueUtils.h"
#include "TestUtils.h"
#include "kitsune/Core/AddrSpace.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"

#include "gtest/gtest.h"

using namespace llvm;

static constexpr StringRef ll = R"(
@g = external global i32

declare void @ext()
declare ptr @get()

define i64 @f(i64 %0) {
  call void @ext()
  %2 = call ptr @get()
  call void %2()
  br label %end

end:
  ret i64 %0
}
)";

namespace {

TEST(KitValueUtils, getName) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);

  GlobalVariable *g = m->getGlobalVariable("g");
  EXPECT_EQ(getName(*g), "g");

  Function *f = m->getFunction("f");
  EXPECT_EQ(getName(*f->getArg(0)), "%0");

  SmallVector<std::string> names;
  for (BasicBlock &bb : *f) {
    names.push_back(getName(bb));
    for (Instruction &inst : bb)
      names.push_back(getName(inst));
  }

  unsigned i = 0;
  EXPECT_EQ(names[i++], "%1");              // The entry basic block is unnamed
  EXPECT_EQ(names[i++], "<call ext>");      // Call does not return a value
  EXPECT_EQ(names[i++], "%2");              // The call returns a value
  EXPECT_EQ(names[i++], "<call %2>");       // The instruction is named
  EXPECT_EQ(names[i++], "<br label %end>"); // Branches have no name
  EXPECT_EQ(names[i++], "end");             // The basic block is named
  EXPECT_EQ(names[i++], "<ret i64 %0>");    // Returns have no name
}

TEST(KitValueUtils, isFalse) {
  LLVMContext ctx;
  Type *i8 = Type::getInt8Ty(ctx);

  EXPECT_TRUE(isFalse(ConstantInt::getFalse(ctx)));
  EXPECT_FALSE(isFalse(ConstantInt::getTrue(ctx)));
  EXPECT_FALSE(isFalse(ConstantInt::get(i8, 0)));
  EXPECT_FALSE(isFalse(ConstantInt::get(i8, 1)));
}

TEST(KitValueUtils, isTrue) {
  LLVMContext ctx;
  Type *i8 = Type::getInt8Ty(ctx);

  EXPECT_TRUE(isTrue(ConstantInt::getTrue(ctx)));
  EXPECT_FALSE(isTrue(ConstantInt::getFalse(ctx)));
  EXPECT_FALSE(isTrue(ConstantInt::get(i8, 0)));
  EXPECT_FALSE(isTrue(ConstantInt::get(i8, 1)));
}

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

TEST(KitValueUtils, isType) {
  LLVMContext ctx;
  Type *i1 = Type::getInt1Ty(ctx);
  Type *i8 = Type::getInt8Ty(ctx);
  Type *i16 = Type::getInt16Ty(ctx);
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);
  Type *f32 = Type::getFloatTy(ctx);
  Type *f64 = Type::getDoubleTy(ctx);

  Constant *ci1 = ConstantInt::get(i1, 0);
  Constant *ci8 = ConstantInt::get(i8, 0);
  Constant *ci16 = ConstantInt::get(i16, 0);
  Constant *ci32 = ConstantInt::get(i32, 0);
  Constant *ci64 = ConstantInt::get(i64, 0);
  Constant *cf32 = ConstantFP::get(f32, 0);
  Constant *cf64 = ConstantFP::get(f64, 0);

  EXPECT_TRUE(isBool(*ci1));
  EXPECT_TRUE(isInt8(*ci8));
  EXPECT_TRUE(isInt16(*ci16));
  EXPECT_TRUE(isInt32(*ci32));
  EXPECT_TRUE(isInt64(*ci64));
  EXPECT_TRUE(isFloat(*cf32));
  EXPECT_TRUE(isDouble(*cf64));

  EXPECT_FALSE(isBool(*ci8));
  EXPECT_FALSE(isInt8(*ci16));
  EXPECT_FALSE(isInt16(*ci32));
  EXPECT_FALSE(isInt32(*ci64));
  EXPECT_FALSE(isInt64(*ci1));
  EXPECT_FALSE(isFloat(*cf64));
  EXPECT_FALSE(isDouble(*cf32));
}

TEST(KitValueUtils, isPointerType) {
  LLVMContext ctx;
  Type *i64 = Type::getInt64Ty(ctx);
  PointerType *pty0 = PointerType::getUnqual(ctx);
  PointerType *pty67 = PointerType::get(ctx, 67);

  Constant *c64 = ConstantInt::get(i64, 0);
  Constant *c0 = ConstantPointerNull::get(pty0);
  Constant *c67 = ConstantPointerNull::get(pty67);

  EXPECT_TRUE(isPointer(*c0));
  EXPECT_TRUE(isPointer(*c67));
  EXPECT_TRUE(isPointer(*c0, 0));
  EXPECT_TRUE(isPointer(*c67, 67));

  EXPECT_FALSE(isPointer(*c0, 67));
  EXPECT_FALSE(isPointer(*c67, 0));

  EXPECT_FALSE(isPointer(*c64));
  EXPECT_FALSE(isPointer(*c64, 0));
  EXPECT_FALSE(isPointer(*c64, 64));
}

TEST(KitValueUtils, getModule) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  Type *i32 = Type::getInt32Ty(ctx);
  FunctionType *funcTy = FunctionType::get(voidTy, {i32}, /*IsVarArg=*/false);
  GlobalValue::LinkageTypes linkage = GlobalValue::ExternalLinkage;

  Constant *c = ConstantInt::get(i32, 11, /*isSigned=*/false);

  Module m("", ctx);

  GlobalVariable *gO = new GlobalVariable(i32, /*isConstant=*/false, linkage);
  GlobalVariable *gM = new GlobalVariable(m, i32, /*isConstant=*/false, linkage,
                                          /*initializer=*/nullptr);

  Function *fO = Function::Create(funcTy, linkage, "fO");
  Function *fM = Function::Create(funcTy, linkage, "fM", &m);

  Argument aO(i32);
  Argument aFO(i32, "", fO);
  Argument aFM(i32, "", fM);

  BasicBlock *bbO = BasicBlock::Create(ctx);
  BasicBlock *bbFO = BasicBlock::Create(ctx, "", fO);
  BasicBlock *bbFM = BasicBlock::Create(ctx, "", fM);

  Instruction *retO = ReturnInst::Create(ctx);
  Instruction *retBBO = ReturnInst::Create(ctx, bbO);
  Instruction *retBBFO = ReturnInst::Create(ctx, bbFO);
  Instruction *retBBFM = ReturnInst::Create(ctx, bbFM);

  EXPECT_FALSE(getModule((Value &)*c));
  EXPECT_FALSE(getModule((Value &)*gO));
  EXPECT_FALSE(getModule((Value &)*fO));
  EXPECT_FALSE(getModule((Value &)aO));
  EXPECT_FALSE(getModule((Value &)aFO));
  EXPECT_FALSE(getModule((Value &)*bbO));
  EXPECT_FALSE(getModule((Value &)*bbFO));
  EXPECT_FALSE(getModule((Value &)*retO));
  EXPECT_FALSE(getModule((Value &)*retBBO));
  EXPECT_FALSE(getModule((Value &)*retBBFO));

  EXPECT_EQ(getModule((Value &)*gM), &m);
  EXPECT_EQ(getModule((Value &)*fM), &m);
  EXPECT_EQ(getModule((Value &)aFM), &m);
  EXPECT_EQ(getModule((Value &)*bbFM), &m);
  EXPECT_EQ(getModule((Value &)*retBBFM), &m);
}

TEST(KitValueUtils, getFunction) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  Type *i32 = Type::getInt32Ty(ctx);
  FunctionType *funcTy = FunctionType::get(voidTy, {i32}, /*IsVarArg=*/false);
  GlobalValue::LinkageTypes linkage = GlobalValue::ExternalLinkage;

  Module m("", ctx);

  Constant *c = ConstantInt::get(i32, 11, /*isSigned=*/false);
  GlobalVariable *g = new GlobalVariable(i32, /*isConstant=*/false, linkage);
  Function *f = Function::Create(funcTy, linkage, "f");

  Argument a(i32);
  Argument af(i32, "", f);

  BasicBlock *bb = BasicBlock::Create(ctx);
  BasicBlock *bbf = BasicBlock::Create(ctx, "", f);

  Instruction *ret = ReturnInst::Create(ctx);
  Instruction *retbb = ReturnInst::Create(ctx, bb);
  Instruction *retbbf = ReturnInst::Create(ctx, bbf);

  EXPECT_FALSE(getFunction((Value &)*c));
  EXPECT_FALSE(getFunction((Value &)*g));
  EXPECT_FALSE(getFunction((Value &)*f));
  EXPECT_FALSE(getFunction((Value &)a));
  EXPECT_FALSE(getFunction((Value &)*bb));
  EXPECT_FALSE(getFunction((Value &)*ret));
  EXPECT_FALSE(getFunction((Value &)*retbb));

  EXPECT_EQ(getFunction((Value &)af), f);
  EXPECT_EQ(getFunction((Value &)*bbf), f);
  EXPECT_EQ(getFunction((Value &)*retbbf), f);
}

TEST(KitValueUtils, getBasicBlock) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  Type *i32 = Type::getInt32Ty(ctx);
  FunctionType *funcTy = FunctionType::get(voidTy, {i32}, /*IsVarArg=*/false);
  GlobalValue::LinkageTypes linkage = GlobalValue::ExternalLinkage;

  Module m("", ctx);

  Constant *c = ConstantInt::get(i32, 11, /*isSigned=*/false);
  GlobalVariable *g = new GlobalVariable(i32, /*isConstant=*/false, linkage);
  Function *f = Function::Create(funcTy, linkage, "f");
  Argument a(i32);

  BasicBlock *bb = BasicBlock::Create(ctx);

  Instruction *ret = ReturnInst::Create(ctx);
  Instruction *retbb = ReturnInst::Create(ctx, bb);

  EXPECT_FALSE(getBasicBlock((Value &)*c));
  EXPECT_FALSE(getBasicBlock((Value &)*g));
  EXPECT_FALSE(getBasicBlock((Value &)*f));
  EXPECT_FALSE(getBasicBlock((Value &)a));
  EXPECT_FALSE(getBasicBlock((Value &)*bb));
  EXPECT_FALSE(getBasicBlock((Value &)*ret));

  EXPECT_EQ(getBasicBlock((Value &)*retbb), bb);
}

TEST(KitValueUtils, getMobileToDefaultCast) {
  LLVMContext ctx;
  Type *ptrD = PointerType::get(ctx, KitAS::Default);
  Type *ptrM = PointerType::get(ctx, KitAS::Mobile);
  Type *ptrO = PointerType::get(ctx, KitAS::Mobile + 1);
  ReturnInst *ret = ReturnInst::Create(ctx);

  Constant *vD = ConstantPointerNull::get(ptrD);
  Constant *vM = ConstantPointerNull::get(ptrM);
  Constant *vO = ConstantPointerNull::get(ptrO);

  Value *cstDToD = CastInst::CreatePointerBitCastOrAddrSpaceCast(vD, ptrD);
  Value *cstDToM = CastInst::CreatePointerBitCastOrAddrSpaceCast(vD, ptrM);
  Value *cstDToO = CastInst::CreatePointerBitCastOrAddrSpaceCast(vD, ptrO);
  Value *cstMToD = CastInst::CreatePointerBitCastOrAddrSpaceCast(vM, ptrD);
  Value *cstMToM = CastInst::CreatePointerBitCastOrAddrSpaceCast(vM, ptrM);
  Value *cstMToO = CastInst::CreatePointerBitCastOrAddrSpaceCast(vM, ptrO);
  Value *cstOToD = CastInst::CreatePointerBitCastOrAddrSpaceCast(vO, ptrD);
  Value *cstOToM = CastInst::CreatePointerBitCastOrAddrSpaceCast(vO, ptrM);
  Value *cstOToO = CastInst::CreatePointerBitCastOrAddrSpaceCast(vO, ptrO);

  EXPECT_FALSE(isMobileToDefaultCast(*ret));
  EXPECT_FALSE(isMobileToDefaultCast(*cstDToD));
  EXPECT_FALSE(isMobileToDefaultCast(*cstDToM));
  EXPECT_FALSE(isMobileToDefaultCast(*cstDToO));
  EXPECT_FALSE(isMobileToDefaultCast(*cstMToM));
  EXPECT_FALSE(isMobileToDefaultCast(*cstMToO));
  EXPECT_FALSE(isMobileToDefaultCast(*cstOToD));
  EXPECT_FALSE(isMobileToDefaultCast(*cstOToM));
  EXPECT_FALSE(isMobileToDefaultCast(*cstOToO));

  EXPECT_TRUE(isMobileToDefaultCast(*cstMToD));
}

} // namespace

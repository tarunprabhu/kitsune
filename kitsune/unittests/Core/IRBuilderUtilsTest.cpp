//===- IRBuilderUtilsTest.cpp - Unit tests for IRBuilder utilities --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/IRBuilderUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

TEST(KitIRBuilderUtils, getFunction) {
  LLVMContext ctx;
  IRBuilder<> builder(ctx);
  Module m("", ctx);
  Type *ret = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(ret, {}, /*isVarArg=*/false);
  Function *o = Function::Create(fty, GlobalValue::InternalLinkage, "orphan");
  Function *f = Function::Create(fty, GlobalValue::InternalLinkage, "f", &m);

  BasicBlock *bb = BasicBlock::Create(ctx, "");
  BasicBlock *bbo = BasicBlock::Create(ctx, "", o);
  BasicBlock *bbf = BasicBlock::Create(ctx, "", f);

  EXPECT_FALSE(getFunction(builder));

  builder.SetInsertPoint(bb);
  EXPECT_FALSE(getFunction(builder));

  builder.SetInsertPoint(bbo);
  EXPECT_EQ(getFunction(builder), o);

  builder.SetInsertPoint(bbf);
  EXPECT_EQ(getFunction(builder), f);
}

TEST(KitIRBuilderUtils, getModule) {
  LLVMContext ctx;
  IRBuilder<> builder(ctx);
  Module m("", ctx);
  Type *ret = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(ret, {}, /*isVarArg=*/false);
  Function *o = Function::Create(fty, GlobalValue::InternalLinkage, "orphan");
  Function *f = Function::Create(fty, GlobalValue::InternalLinkage, "f", &m);

  BasicBlock *bb = BasicBlock::Create(ctx, "");
  BasicBlock *bbo = BasicBlock::Create(ctx, "", o);
  BasicBlock *bbf = BasicBlock::Create(ctx, "", f);

  EXPECT_FALSE(getModule(builder));

  builder.SetInsertPoint(bb);
  EXPECT_FALSE(getModule(builder));

  builder.SetInsertPoint(bbo);
  EXPECT_FALSE(getModule(builder));

  builder.SetInsertPoint(bbf);
  EXPECT_EQ(getModule(builder), &m);
}

TEST(KitIRBuilderUtils, createLibFuncCall) {
  LLVMContext ctx;
  Module m("", ctx);
  Type *ret = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(ret, {}, /*isVarArg=*/false);
  Function *f = Function::Create(fty, GlobalValue::InternalLinkage, "f", &m);
  BasicBlock *bb = BasicBlock::Create(ctx, "", f);
  Value *zero = ConstantInt::get(Type::getInt64Ty(ctx), 0, /*isSigned=*/false);

  KitFunc libFunc = KitFunc::kitrt_malloc;

  IRBuilder<> builder(ctx);
  builder.SetInsertPoint(bb);
  Value *call = createCall(builder, libFunc, {zero}, "nuuk");

  EXPECT_EQ(bb->size(), 1U);
  EXPECT_EQ(&*bb->begin(), call);
  EXPECT_TRUE(isa<CallInst>(call));
  EXPECT_EQ(cast<CallInst>(call)->getCalledFunction()->getName(),
            getLibFuncName(libFunc));
  EXPECT_EQ(call->getName(), "nuuk");
}

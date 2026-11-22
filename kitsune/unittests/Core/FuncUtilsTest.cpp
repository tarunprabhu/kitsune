//===- FuncUtilsTest.cpp - Unit tests for Kitsune's function utilities ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/FuncUtils.h"
#include "TestUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitFuncUtils, getName) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(voidTy, {}, /*IsVarArg=*/false);
  Module m("", ctx);
  Function *f = cast<Function>(m.getOrInsertFunction("", fty).getCallee());

  EXPECT_FALSE(f->hasName());
  EXPECT_EQ(getName(*f), "@0");
}

TEST(KitFuncUtils, copyAttrsFunc) {
  LLVMContext ctx;
  Module m("", ctx);

  PointerType *ptrTy = PointerType::getUnqual(ctx);
  FunctionType *fty = FunctionType::get(ptrTy, {ptrTy}, /*IsVarArg=*/false);
  GlobalValue::LinkageTypes linkage = GlobalValue::ExternalLinkage;
  Constant *cnull = ConstantPointerNull::get(ptrTy);

  Function *origF = Function::Create(fty, linkage, "original", &m);
  origF->addRetAttr(Attribute::NoAlias);
  origF->getArg(0)->addAttr(Attribute::NonNull);
  origF->addFnAttr(Attribute::NoUnwind);

  Function *dup1 = Function::Create(fty, linkage, "dup1", &m);
  copyAttrs(*dup1, *origF);

  EXPECT_TRUE(dup1->hasRetAttribute(Attribute::NoAlias));
  EXPECT_TRUE(dup1->hasFnAttribute(Attribute::NoUnwind));
  EXPECT_FALSE(dup1->hasParamAttribute(0, Attribute::NoAlias));
  EXPECT_FALSE(dup1->hasGC());
  EXPECT_FALSE(dup1->hasPersonalityFn());
  EXPECT_FALSE(dup1->hasPrefixData());
  EXPECT_FALSE(dup1->hasPrologueData());

  origF->setGC("fancy-gc");
  origF->setPersonalityFn(cnull);
  origF->setPrefixData(cnull);
  origF->setPrologueData(cnull);
  copyAttrs(*dup1, *origF);
  EXPECT_TRUE(dup1->hasRetAttribute(Attribute::NoAlias));
  EXPECT_TRUE(dup1->hasFnAttribute(Attribute::NoUnwind));
  EXPECT_FALSE(dup1->hasParamAttribute(0, Attribute::NoAlias));
  EXPECT_TRUE(dup1->hasGC());
  EXPECT_EQ(dup1->getGC(), "fancy-gc");
  EXPECT_TRUE(dup1->hasPersonalityFn());
  EXPECT_EQ(dup1->getPersonalityFn(), cnull);
  EXPECT_TRUE(dup1->hasPrefixData());
  EXPECT_EQ(dup1->getPrefixData(), cnull);
  EXPECT_TRUE(dup1->hasPrologueData());
  EXPECT_EQ(dup1->getPrologueData(), cnull);
}

TEST(KitFuncUtils, copyAttrsArgs) {
  LLVMContext ctx;
  Module m("", ctx);

  PointerType *ptrTy = PointerType::getUnqual(ctx);
  FunctionType *fty =
      FunctionType::get(ptrTy, {ptrTy, ptrTy}, /*IsVarArg=*/false);
  GlobalValue::LinkageTypes linkage = GlobalValue::ExternalLinkage;

  Function *origF = Function::Create(fty, linkage, "original", &m);
  origF->addRetAttr(Attribute::NoAlias);
  origF->getArg(0)->addAttr(Attribute::NonNull);
  origF->getArg(1)->addAttr(Attribute::NoUndef);

  Function *dup1 = Function::Create(fty, linkage, "dup1", &m);

  copyAttrs(*dup1->getArg(0), *origF->getArg(0));
  EXPECT_TRUE(dup1->hasParamAttribute(0, Attribute::NonNull));
  EXPECT_FALSE(dup1->hasParamAttribute(1, Attribute::NoUndef));
  EXPECT_FALSE(dup1->hasRetAttribute(Attribute::NoAlias));

  copyAttrs(*dup1->getArg(1), *origF->getArg(1));
  EXPECT_TRUE(dup1->hasParamAttribute(0, Attribute::NonNull));
  EXPECT_TRUE(dup1->hasParamAttribute(1, Attribute::NoUndef));
  EXPECT_FALSE(dup1->hasRetAttribute(Attribute::NoAlias));
}

TEST(KitFuncUtils, getAttrAsString) {
  LLVMContext ctx;
  Type *ret = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(ret, {}, /*IsVarArg=*/false);
  Function *f = Function::Create(fty, GlobalValue::InternalLinkage, "f");
  f->addFnAttr("attr", "val");
  f->addFnAttr(Attribute::WillReturn);
  f->setUWTableKind(UWTableKind::Sync);

  EXPECT_FALSE(getAttrAsString(*f, "non-existent"));
  EXPECT_FALSE(getAttrAsString(*f, "willreturn"));
  EXPECT_FALSE(getAttrAsString(*f, "uwtable"));

  EXPECT_EQ(*getAttrAsString(*f, "attr"), "val");
}

TEST(KitFuncUtils, getBlockNamed) {
  LLVMContext ctx;
  Type *ret = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(ret, {}, /*IsVarArg=*/false);
  Function *f = Function::Create(fty, GlobalValue::InternalLinkage, "f");

  BasicBlock *entry = BasicBlock::Create(ctx, "entry", f);
  [[maybe_unused]] BasicBlock *body = BasicBlock::Create(ctx, "", f);
  BasicBlock *exit = BasicBlock::Create(ctx, "exit", f);

  EXPECT_EQ(getBlockNamed("entry", *f), entry);
  EXPECT_EQ(getBlockNamed("exit", *f), exit);

  EXPECT_FALSE(getBlockNamed("", *f));
  EXPECT_FALSE(getBlockNamed("body", *f));
}

static constexpr StringRef calls = R"(
define ptr addrspace(67) @f1(i64 %n) {
  %buf1 = call ptr addrspace(67) @llvm.kit.mobile.alloc(i32 1, i64 %n)
  ret ptr addrspace(67) %buf1
}

define void @f2(i64 %n) {
  %ptr2 = call ptr addrspace(67) @llvm.kit.mobile.alloc(i32 2, i64 %n)
  %ptr4 = call ptr addrspace(67) @llvm.kit.mobile.alloc(i32 4, i64 %n)
  ret void
}

define void @f3(ptr addrspace(67) %buf) {
  call void @llvm.kit.mobile.free(i32 1, ptr addrspace(67) %buf)
  ret void
}
)";
TEST(KitFuncUtils, getCalls) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);

  std::unique_ptr<Module> m = parseIR(ctx, calls);
  Constant *one = ConstantInt::get(i32, 1, /*isSigned=*/false);
  Constant *two = ConstantInt::get(i32, 2, /*isSigned=*/false);
  Constant *four = ConstantInt::get(i32, 4, /*isSigned=*/false);

  Function *f1 = m->getFunction("f1");
  SmallVector<CallInst *, 0> calls1 =
      getCalls(*f1, Intrinsic::kit_cpu_num_threads);
  EXPECT_EQ(calls1.size(), 0U);

  Function *f2 = m->getFunction("f2");
  SmallVector<CallInst *, 0> calls2 =
      getCalls(*f2, Intrinsic::kit_mobile_alloc);
  EXPECT_EQ(calls2.size(), 2U);
  EXPECT_EQ(calls2[0]->getArgOperand(0), two);
  EXPECT_EQ(calls2[1]->getArgOperand(0), four);

  Function *f3 = m->getFunction("f3");
  SmallVector<CallInst *, 0> calls3 = getCalls(*f3, Intrinsic::kit_mobile_free);
  EXPECT_EQ(calls3.size(), 1U);
  EXPECT_EQ(calls3[0]->getArgOperand(0), one);
}

} // namespace

//==- FunctionUtilsTest.cpp - Unit tests for Kitsune's function utilities --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/FunctionUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

TEST(FunctionUtilsTest, copyAttrsFunc) {
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

TEST(FunctionUtilsTest, copyAttrsArgs) {
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

//===- ArgUtilsTest.cpp - Unit tests for Kitsune's argument utilities -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ArgUtils.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitArgUtils, getModule) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  Type *i32 = Type::getInt32Ty(ctx);
  FunctionType *funcTy = FunctionType::get(voidTy, {i32}, /*IsVarArg=*/false);
  GlobalValue::LinkageTypes linkage = GlobalValue::ExternalLinkage;

  Module m("", ctx);
  Function *fO = Function::Create(funcTy, linkage, "fO");
  Function *fM = Function::Create(funcTy, linkage, "fM", &m);

  Argument aO(i32);
  Argument aFO(i32, "", fO);
  Argument aFM(i32, "", fM);

  EXPECT_FALSE(getModule(aO));
  EXPECT_FALSE(getModule(aFO));
  EXPECT_EQ(getModule(aFM), &m);
}

TEST(KitArgUtils, getName) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  Type *i32Ty = Type::getInt32Ty(ctx);
  FunctionType *fty =
      FunctionType::get(voidTy, {i32Ty, i32Ty}, /*IsVarArg=*/false);
  Module m("", ctx);
  Function *f = cast<Function>(m.getOrInsertFunction("", fty).getCallee());
  Argument *arg0 = f->getArg(0);
  Argument *arg1 = f->getArg(1);

  arg0->setName("arg0");

  EXPECT_TRUE(arg0->hasName());
  EXPECT_EQ(getName(*arg0), "arg0");

  EXPECT_FALSE(arg1->hasName());
  EXPECT_EQ(getName(*arg1), "%0");
}

} // namespace

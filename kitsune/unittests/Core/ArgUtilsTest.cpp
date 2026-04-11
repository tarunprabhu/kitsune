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

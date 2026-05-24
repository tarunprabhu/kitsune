//===- ValueUtilsTest.cpp - Tests for Kitsune's value utilities -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ValueUtils.h"
#include "TestUtils.h"
#include "llvm/IR/Constants.h"
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

} // namespace

//===- UseUtilsTest.cpp - Tests for Kitsune's LLVM Use utilities ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/UseUtils.h"
#include "TestUtils.h"
// #include "llvm/IR/LLVMContext.h"

#include "gtest/gtest.h"

using namespace llvm;

static constexpr StringRef ll = R"(
@g = global ptr @ext

declare i32 @ext(i32)

define i32 @f(i32 %a, i32 %b) {
entry:
  call void @ext(i32 %a)
  br label %exit

exit:
  ret i32 %b
}
)";

namespace {

TEST(KitUseUtils, isUseInBlock) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  Argument *a = f->getArg(0);
  Argument *b = f->getArg(1);
  BasicBlock *entry = &f->getEntryBlock();
  BasicBlock *exit = entry->getTerminator()->getSuccessor(0);

  Use &useA = *a->use_begin();
  Use &useB = *b->use_begin();

  EXPECT_TRUE(isUseInBlock(useA, *entry));
  EXPECT_FALSE(isUseInBlock(useA, *exit));
  EXPECT_FALSE(isUseInBlock(useB, *entry));
  EXPECT_TRUE(isUseInBlock(useB, *exit));
}

TEST(KitUseUtils, isUseInConstant) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *ext = m->getFunction("ext");

  for (Use &use : ext->uses())
    EXPECT_EQ(isUseInConstant(use), isa<Constant>(use.getUser()));
}

} // namespace

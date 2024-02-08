//==- GVUtilsTest.cpp - Unit tests for Kitsune's global variable utilities -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/GVUtils.h"
#include "TestUtils.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

static constexpr StringRef ll = R"(
@g = external global i32
@0 = constant i32 19
)";

namespace {

TEST(KitGVUtils, getName) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);

  for (GlobalVariable &g : m->globals())
    if (g.hasName())
      EXPECT_EQ(getName(g), "g");
    else
      EXPECT_EQ(getName(g), "@0");
}

} // namespace

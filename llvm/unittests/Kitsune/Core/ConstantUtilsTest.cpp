//===- ConstantUtilsTest.cpp - Unit tests for constant utilities ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ConstantUtils.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

static std::unique_ptr<Module> parseIR(LLVMContext &ctx, StringRef ir) {
  SMDiagnostic err;
  std::unique_ptr<Module> m = parseAssemblyString(ir, err, ctx);
  if (!m)
    err.print("parseIR", errs());
  return m;
}

TEST(KitConstantUtils, createConstIntTT) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  ConstantInt *cCuda = createConstInt(TTID::Cuda, ctx);

  EXPECT_EQ(cCuda->getType(), i32);
  EXPECT_EQ(cCuda->getZExtValue(), int(TTID::Cuda));
}

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
  EXPECT_EQ(createConstString(s1, *m, "newName"), g1);
  EXPECT_FALSE(createConstString(s1, *m, "newName")->hasName());

  // When asking for a different string, make sure that works too.
  GlobalVariable *g2 = createConstString(s2, *m, "s2");

  EXPECT_EQ(m->global_size(), 3U);
  EXPECT_EQ(g2->getName(), "s2");
}

} // namespace

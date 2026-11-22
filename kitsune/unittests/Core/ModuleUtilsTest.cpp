//===- ModuleUtilsTest.cpp - Unit tests for module utilities --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ModuleUtils.h"
#include "TestUtils.h"
#include "kitsune/Core/ModuleAttrs.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitModuleUtils, deviceModuleMetadata) {
  LLVMContext ctx;
  Module m("strathclyde", ctx);

  addDeviceModuleFlagsAttr(m, TTID::Serial);
  EXPECT_TRUE(hasDeviceModuleFlagsAttr(m));
  EXPECT_EQ(getTTIDFromDeviceModuleFlagsAttr(m), TTID::Serial);
  EXPECT_EQ(getNameFromDeviceModuleFlagsAttr(m), "strathclyde");
}

TEST(KitModuleUtils, getOrInsertFunction) {
  LLVMContext ctx;
  Module m("", ctx);

  Type *ptr = PointerType::getUnqual(ctx);
  Type *i32 = Type::getInt32Ty(ctx);
  FunctionType *fty = FunctionType::get(ptr, {i32}, /*isVarArg=*/false);

  EXPECT_FALSE(m.getFunction("f"));
  Function *f = getOrInsertFunction(m, "f", fty);
  EXPECT_EQ(f->getName(), "f");
  EXPECT_EQ(f->arg_size(), 1U);
  EXPECT_EQ(f->getReturnType(), ptr);
  EXPECT_EQ(f->getArg(0)->getType(), i32);
  EXPECT_EQ(getOrInsertFunction(m, "f", fty), f);

  EXPECT_FALSE(m.getFunction("g"));
  Function *g = getOrInsertFunction(m, "g", i32, ptr);
  EXPECT_EQ(g->getName(), "g");
  EXPECT_EQ(g->arg_size(), 1U);
  EXPECT_EQ(g->getReturnType(), i32);
  EXPECT_EQ(g->getArg(0)->getType(), ptr);
  EXPECT_EQ(getOrInsertFunction(m, "g", fty), g);
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

define void @ffree(ptr addrspace(67) %buf) {
  call void @llvm.kit.mobile.free(i32 1, ptr addrspace(67) %buf)
  ret void
}
)";
TEST(KitModuleUtils, getCalls) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);

  std::unique_ptr<Module> m = parseIR(ctx, calls);
  Constant *one = ConstantInt::get(i32, 1, /*isSigned=*/false);
  Constant *two = ConstantInt::get(i32, 2, /*isSigned=*/false);
  Constant *four = ConstantInt::get(i32, 4, /*isSigned=*/false);

  SmallVector<CallInst *, 0> thrds =
      getCalls(*m, Intrinsic::kit_cpu_num_threads);
  EXPECT_EQ(thrds.size(), 0U);

  SmallVector<CallInst *, 0> frees = getCalls(*m, Intrinsic::kit_mobile_free);
  EXPECT_EQ(frees.size(), 1U);
  EXPECT_EQ(frees[0]->getArgOperand(0), one);

  SmallVector<CallInst *, 0> allocs = getCalls(*m, Intrinsic::kit_mobile_alloc);
  EXPECT_EQ(allocs.size(), 3U);
  EXPECT_EQ(allocs[0]->getArgOperand(0), one);
  EXPECT_EQ(allocs[1]->getArgOperand(0), two);
  EXPECT_EQ(allocs[2]->getArgOperand(0), four);
}

} // namespace

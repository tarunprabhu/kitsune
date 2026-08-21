//===- ModuleUtilsTest.cpp - Unit tests for module utilities --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Core/ModuleAttrs.h"
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

} // namespace

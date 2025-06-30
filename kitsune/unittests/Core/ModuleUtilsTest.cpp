//===- ModuleUtilsTest.cpp - Unit tests for module utilities --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ModuleUtils.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitModuleUtils, setAndCheckModuleMD) {
  LLVMContext ctx;
  Module m("modname", ctx);

  EXPECT_FALSE(hasDeviceModuleMetadata(m));
  EXPECT_FALSE(getTTIDFromDeviceModuleMetadata(m));
  EXPECT_FALSE(getNameFromDeviceModuleMetadata(m));

  addDeviceModuleMetadata(TTID::Serial, m);
  EXPECT_TRUE(hasDeviceModuleMetadata(m));
  EXPECT_EQ(*getTTIDFromDeviceModuleMetadata(m), TTID::Serial);
  EXPECT_EQ(*getNameFromDeviceModuleMetadata(m), "modname");
}

} // namespace

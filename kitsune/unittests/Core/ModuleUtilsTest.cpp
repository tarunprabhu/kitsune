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

} // namespace

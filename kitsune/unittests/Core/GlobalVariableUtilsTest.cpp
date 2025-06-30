//=- GlobalVariableUtilsTest.cpp - Unit tests for global variable utilities -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/GlobalVariableUtils.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitGlobalVariableUtils, getAttrValueAsTTID) {
  LLVMContext ctx;
  Type *i64 = Type::getInt64Ty(ctx);
  GlobalVariable g(i64, false, GlobalValue::ExternalLinkage);

  EXPECT_EQ(getAttrValueAsTTID(g, Attribute::KitTT), std::nullopt);

  g.addAttribute(Attribute::getWithTTID(ctx, TTID::Cuda));

  EXPECT_EQ(getAttrValueAsTTID(g, Attribute::KitTT), TTID::Cuda);
}

} // namespace

//===- StringUtilsTest.cpp - Unit tests for tapir target properties -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Support/StringUtils.h"

#include "gtest/gtest.h"

using namespace llvm;

TEST(StringUtils, sjoin) {
  EXPECT_EQ(sjoin(""), "");
  EXPECT_EQ(sjoin('x'), "x");
  EXPECT_EQ(sjoin("a", "b"), "ab");
  EXPECT_EQ(sjoin("a", "bc", "def"), "abcdef");
  EXPECT_EQ(sjoin(1, 2, 3), "123");
  EXPECT_EQ(sjoin("(", 1, ",", 2, ")"), "(1,2)");
}

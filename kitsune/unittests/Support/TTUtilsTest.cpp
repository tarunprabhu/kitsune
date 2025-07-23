//===- TTUtilsTest.cpp - Unit tests for tapir target properties -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Support/TTUtils.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(TTUtilsTest, ttsAll) {
  EXPECT_EQ(unsigned(TTID::Nolo), 0);
  EXPECT_EQ(unsigned(TTID::Serial), 1);

  EXPECT_EQ(ArrayRef<TTID>(ttsAll).size(), 10U);
}

TEST(TTUtilsTest, ttsUsingEmbBC) {
  ArrayRef<TTID> tts = ttsUsingEmbBC;

  EXPECT_EQ(ArrayRef<TTID>(tts).size(), 2U);
  EXPECT_EQ(tts[0], TTID::Cuda);
  EXPECT_EQ(tts[1], TTID::Hip);
}

TEST(TTUtilsTest, ttUsesEmbBC) {
  EXPECT_FALSE(ttUsesEmbBC(TTID::Nolo));
  EXPECT_FALSE(ttUsesEmbBC(TTID::Serial));
  EXPECT_FALSE(ttUsesEmbBC(TTID::OpenCilk));

  EXPECT_TRUE(ttUsesEmbBC(TTID::Cuda));
  EXPECT_TRUE(ttUsesEmbBC(TTID::Hip));
}

} // namespace

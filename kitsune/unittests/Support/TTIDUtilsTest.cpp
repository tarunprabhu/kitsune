//===- TTIDUtilsTest.cpp - Unit tests for TTID utilities ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Support/TTIDUtils.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(TTUtilsTest, ttsGenEmbBC) {
  ArrayRef<TTID> tts = ttsGenEmbBC();

  EXPECT_EQ(tts.size(), 2U);
  EXPECT_EQ(tts[0], TTID::Cuda);
  EXPECT_EQ(tts[1], TTID::Hip);
}

TEST(TTUtilsTest, doesTTGenerateEmbBC) {
  EXPECT_FALSE(doesTTGenEmbBC(TTID::Nolo));
  EXPECT_FALSE(doesTTGenEmbBC(TTID::Serial));
  EXPECT_FALSE(doesTTGenEmbBC(TTID::OpenCilk));

  EXPECT_TRUE(doesTTGenEmbBC(TTID::Cuda));
  EXPECT_TRUE(doesTTGenEmbBC(TTID::Hip));
}

TEST(TTUtilsTest, isGPUTT) {
  EXPECT_TRUE(isGPUTT(TTID::Cuda));
  EXPECT_TRUE(isGPUTT(TTID::Hip));

  EXPECT_FALSE(isGPUTT(TTID::Nolo));
  EXPECT_FALSE(isGPUTT(TTID::Serial));
  EXPECT_FALSE(isGPUTT(TTID::OpenCilk));
  EXPECT_FALSE(isGPUTT(TTID::Pthreads));
  EXPECT_FALSE(isGPUTT(TTID::Qthreads));
}

} // namespace

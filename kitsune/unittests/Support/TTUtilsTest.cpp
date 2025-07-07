//===- TTUtilsTest.cpp - Unit tests for tapir target properties -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Support/TTUtils.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(TTUtilsTest, ttsAll) {
  ArrayRef<TTID> tts = ttsAll();

  EXPECT_EQ(tts.size(), 10U);
}

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

} // namespace

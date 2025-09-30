//===- PipelineUtilsTest.cpp - Tests for Kitsune's pipeline utilities  ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Passes/PipelineUtils.h"

#include "gtest/gtest.h"

using namespace llvm;

TEST(PipelineUtils, isKitsuneOrTapirPipelineAlias) {
  EXPECT_TRUE(isKitsuneOrTapirPipelineAlias("tapir-lowering"));
  EXPECT_TRUE(isKitsuneOrTapirPipelineAlias("tapir-lowering-loops"));
  EXPECT_TRUE(isKitsuneOrTapirPipelineAlias("kit-lowering"));

  EXPECT_TRUE(isKitsuneOrTapirPipelineAlias("tapir-lowering<O1>"));
  EXPECT_TRUE(isKitsuneOrTapirPipelineAlias("tapir-lowering-loops<O2>"));
  EXPECT_TRUE(isKitsuneOrTapirPipelineAlias("kit-lowering<O3>"));

  // These are obviously not how the pipeline would ever appear, but they pass
  // anyway. This is fine because this function only checks if the argument is
  // "more-or-less" as expected.
  EXPECT_TRUE(isKitsuneOrTapirPipelineAlias("tapir-lowering<"));
  EXPECT_TRUE(isKitsuneOrTapirPipelineAlias("tapir-lowering-loops<"));
  EXPECT_TRUE(isKitsuneOrTapirPipelineAlias("kit-lowering<"));

  // Currently, there is no kit-lowering-loops pipeline.
  EXPECT_FALSE(isKitsuneOrTapirPipelineAlias("kit-lowering-loops"));
  EXPECT_FALSE(isKitsuneOrTapirPipelineAlias("kit-lowering-loops<O3>"));
  EXPECT_FALSE(isKitsuneOrTapirPipelineAlias("kit-lowering-loops<"));

  EXPECT_FALSE(isKitsuneOrTapirPipelineAlias(""));
  EXPECT_FALSE(isKitsuneOrTapirPipelineAlias("kit-lowering>"));
  EXPECT_FALSE(isKitsuneOrTapirPipelineAlias("tapir-loops-lowering"));
  EXPECT_FALSE(isKitsuneOrTapirPipelineAlias("tapir-lowering-O3"));
}

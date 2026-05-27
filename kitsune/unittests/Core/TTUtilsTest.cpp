//===- TTUtilsTest.cpp - Unit tests for tapir target utilities ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/TTUtils.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitTTUtils, generatesEmbBC) {
  EXPECT_FALSE(generatesEmbBC(TTID::Nolo));
  EXPECT_FALSE(generatesEmbBC(TTID::Serial));
  EXPECT_FALSE(generatesEmbBC(TTID::OpenCilk));

  EXPECT_TRUE(generatesEmbBC(TTID::Cuda));
  EXPECT_TRUE(generatesEmbBC(TTID::Hip));
}

TEST(KitTTUtils, isGPUTT) {
  EXPECT_TRUE(isGPUTT(TTID::Cuda));
  EXPECT_TRUE(isGPUTT(TTID::Hip));

  EXPECT_FALSE(isGPUTT(TTID::Nolo));
  EXPECT_FALSE(isGPUTT(TTID::Serial));
  EXPECT_FALSE(isGPUTT(TTID::OpenCilk));
  EXPECT_FALSE(isGPUTT(TTID::Pthreads));
  EXPECT_FALSE(isGPUTT(TTID::Qthreads));
}

} // namespace

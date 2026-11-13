//===- ReductionStrategyTest.cpp - Unit tests for reduction strategy ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ReductionStrategy.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitReductionStrategy, fromInt) {
  EXPECT_EQ(fromInt<ReductionStrategy>(0), ReductionStrategy::Auto);
  EXPECT_EQ(fromInt<ReductionStrategy>(1), ReductionStrategy::Naive);
  EXPECT_EQ(fromInt<ReductionStrategy>(2), ReductionStrategy::Direct);
  EXPECT_EQ(fromInt<ReductionStrategy>(6), ReductionStrategy::GPUSharedMemory);
  EXPECT_EQ(fromInt<ReductionStrategy>(10), ReductionStrategy::GPUWarpShuffle);
  EXPECT_EQ(fromInt<ReductionStrategy>(14),
            ReductionStrategy::GPUWarpShuffleWithSharedMemory);
}

TEST(KitReductionStrategy, fromString) {
  EXPECT_EQ(fromString<ReductionStrategy>("auto"), ReductionStrategy::Auto);
  EXPECT_EQ(fromString<ReductionStrategy>("naive"), ReductionStrategy::Naive);
  EXPECT_EQ(fromString<ReductionStrategy>("direct"), ReductionStrategy::Direct);
  EXPECT_EQ(fromString<ReductionStrategy>("shmem"),
            ReductionStrategy::GPUSharedMemory);
  EXPECT_EQ(fromString<ReductionStrategy>("wshfl"),
            ReductionStrategy::GPUWarpShuffle);
  EXPECT_EQ(fromString<ReductionStrategy>("wshflshmem"),
            ReductionStrategy::GPUWarpShuffleWithSharedMemory);
}

TEST(KitReductionStrategy, isCPUStrategy) {
  EXPECT_TRUE(isCPUStrategy(ReductionStrategy::Auto));
  EXPECT_TRUE(isCPUStrategy(ReductionStrategy::Naive));
  EXPECT_TRUE(isCPUStrategy(ReductionStrategy::Direct));

  EXPECT_FALSE(isCPUStrategy(ReductionStrategy::GPUSharedMemory));
  EXPECT_FALSE(isCPUStrategy(ReductionStrategy::GPUWarpShuffle));
  EXPECT_FALSE(
      isCPUStrategy(ReductionStrategy::GPUWarpShuffleWithSharedMemory));
}

TEST(KitReductionStrategy, isGPUStrategy) {
  EXPECT_TRUE(isGPUStrategy(ReductionStrategy::Auto));
  EXPECT_TRUE(isGPUStrategy(ReductionStrategy::Naive));
  EXPECT_TRUE(isGPUStrategy(ReductionStrategy::Direct));
  EXPECT_TRUE(isGPUStrategy(ReductionStrategy::GPUSharedMemory));
  EXPECT_TRUE(isGPUStrategy(ReductionStrategy::GPUWarpShuffle));
  EXPECT_TRUE(isGPUStrategy(ReductionStrategy::GPUWarpShuffleWithSharedMemory));
}

} // namespace

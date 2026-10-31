//===- MiscUtilsTest.cpp - Tests for miscellaneous utilities --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common/env.h"
#include "common/utils.h"

#include "llvm/ADT/ArrayRef.h"

#include "gtest/gtest.h"

using namespace kitrt;

namespace {

TEST(KitrtUtils, num_cpus) {
  // We can only check that we always return at least 1. There isn't any
  // advantage to trying to check that the "correct" value for the number of
  // CPU's is returned because the implementation simply calls
  // std::thread::hardware_concurrency.
  EXPECT_GE(getNumCPUs(), 1U);
}

TEST(KitrtUtils, num_threads) {
  uint32_t cpus = getNumCPUs();

  envUnset("KIT_NUM_THREADS");
  EXPECT_EQ(getNumThreadsOrCPUs(), cpus);

  envSet("KIT_NUM_THREADS", 41U);
  envSet("ALTERNATIVE", 97U);
  EXPECT_EQ(getNumThreadsOrCPUs(), 41U);
  EXPECT_EQ(getNumThreadsOrCPUs("ALTERNATIVE"), 41U);

  envUnset("KIT_NUM_THREADS");
  EXPECT_EQ(getNumThreadsOrCPUs(), cpus);
  EXPECT_EQ(getNumThreadsOrCPUs("ALTERNATIVE"), 97U);

  envUnset("ALTERNATIVE");
  EXPECT_EQ(getNumThreadsOrCPUs("KIT_NUM_THREADS"), cpus);
  EXPECT_EQ(getNumThreadsOrCPUs("ALTERNATIVE"), cpus);

  envSet("KIT_NUM_THREADS", "forty-one");
  envSet("ALTERNATIVE", "ninety-seven");
  EXPECT_EQ(getNumThreadsOrCPUs(), cpus);
  EXPECT_EQ(getNumThreadsOrCPUs("ALTERNATIVE"), cpus);

  envSet("KIT_NUM_THREADS", 0U);
  envSet("ALTERNATIVE", 11U);
  EXPECT_EQ(getNumThreadsOrCPUs(), cpus);
  EXPECT_EQ(getNumThreadsOrCPUs("ALTERNATIVE"), 11U);

  envSet("KIT_NUM_THREADS", "0");
  envSet("ALTERNATIVE", 0U);
  EXPECT_EQ(getNumThreadsOrCPUs("ALTERNATIVE"), cpus);

  envUnset("KIT_NUM_THREADS");
  envUnset("ALTERNATIVE");
}

TEST(KitrtUtils, nearestPowerOf2LE) {
  auto pow2 = [](uint32_t exp) -> uint32_t { return 1U << (exp - 1); };

  EXPECT_EQ(nearestPowerOf2LE(0), 0U);
  EXPECT_EQ(nearestPowerOf2LE(1), 1U);
  EXPECT_EQ(nearestPowerOf2LE(2), 2U);
  EXPECT_EQ(nearestPowerOf2LE(3), 2U);
  EXPECT_EQ(nearestPowerOf2LE(4), 4U);
  EXPECT_EQ(nearestPowerOf2LE(5), 4U);
  EXPECT_EQ(nearestPowerOf2LE(6), 4U);
  EXPECT_EQ(nearestPowerOf2LE(7), 4U);
  EXPECT_EQ(nearestPowerOf2LE(8), 8U);
  EXPECT_EQ(nearestPowerOf2LE(9), 8U);
  EXPECT_EQ(nearestPowerOf2LE(15), 8U);
  EXPECT_EQ(nearestPowerOf2LE(16), 16U);
  EXPECT_EQ(nearestPowerOf2LE(17), 16U);
  EXPECT_EQ(nearestPowerOf2LE(pow2(31)), 1U << 30);
  EXPECT_EQ(nearestPowerOf2LE(pow2(32) - 1), 1U << 30);
  EXPECT_EQ(nearestPowerOf2LE(pow2(32)), 1U << 31);
}

TEST(KitrtUtils, nearestMultipleLT) {
  EXPECT_EQ(nearestMultipleLT(0, 99), 0U);
  EXPECT_EQ(nearestMultipleLT(99, 0), 99U);
  EXPECT_EQ(nearestMultipleLT(1, 99), 98U);
  EXPECT_EQ(nearestMultipleLT(2, 99), 98U);
  EXPECT_EQ(nearestMultipleLT(3, 99), 96U);
  EXPECT_EQ(nearestMultipleLT(32, 99), 96U);
  EXPECT_EQ(nearestMultipleLT(33, 99), 66U);
  EXPECT_EQ(nearestMultipleLT(34, 99), 68U);
  EXPECT_EQ(nearestMultipleLT(49, 99), 98U);
  EXPECT_EQ(nearestMultipleLT(50, 99), 50U);
  EXPECT_EQ(nearestMultipleLT(51, 99), 51U);
  EXPECT_EQ(nearestMultipleLT(98, 99), 98U);
  EXPECT_EQ(nearestMultipleLT(99, 99), 99U);
  EXPECT_EQ(nearestMultipleLT(100, 99), 100U);
  EXPECT_EQ(nearestMultipleLT(101, 99), 101U);
  EXPECT_EQ(nearestMultipleLT(197, 99), 197U);
  EXPECT_EQ(nearestMultipleLT(198, 99), 198U);
  EXPECT_EQ(nearestMultipleLT(199, 99), 199U);
}

TEST(KitrtUtils, ctx) {
  using NestThreads = std::vector<uint32_t>;

  auto vec = [](uint32_t t0, uint32_t t1, uint32_t t2) -> NestThreads {
    return {t0, t1, t2};
  };

  auto ctxThreads = [](uint64_t totalThreads, uint32_t depth) -> NestThreads {
    LoopNestContext *ctx = nestBegin(totalThreads, depth);
    NestThreads threads = llvm::ArrayRef<uint32_t>(ctx->threads).vec();
    nestEnd(ctx);

    return threads;
  };

  EXPECT_EQ(ctxThreads(1, 1), vec(1, 1, 1));
  EXPECT_EQ(ctxThreads(2, 1), vec(2, 1, 1));
  EXPECT_EQ(ctxThreads(3, 1), vec(3, 1, 1));
  EXPECT_EQ(ctxThreads(4, 1), vec(4, 1, 1));
  EXPECT_EQ(ctxThreads(5, 1), vec(5, 1, 1));
  EXPECT_EQ(ctxThreads(6, 1), vec(6, 1, 1));
  EXPECT_EQ(ctxThreads(7, 1), vec(7, 1, 1));
  EXPECT_EQ(ctxThreads(8, 1), vec(8, 1, 1));
  EXPECT_EQ(ctxThreads(12, 1), vec(12, 1, 1));
  EXPECT_EQ(ctxThreads(16, 1), vec(16, 1, 1));
  EXPECT_EQ(ctxThreads(20, 1), vec(20, 1, 1));
  EXPECT_EQ(ctxThreads(24, 1), vec(24, 1, 1));
  EXPECT_EQ(ctxThreads(32, 1), vec(32, 1, 1));
  EXPECT_EQ(ctxThreads(36, 1), vec(36, 1, 1));
  EXPECT_EQ(ctxThreads(40, 1), vec(40, 1, 1));
  EXPECT_EQ(ctxThreads(48, 1), vec(48, 1, 1));
  EXPECT_EQ(ctxThreads(64, 1), vec(64, 1, 1));
  EXPECT_EQ(ctxThreads(72, 1), vec(72, 1, 1));
  EXPECT_EQ(ctxThreads(80, 1), vec(80, 1, 1));
  EXPECT_EQ(ctxThreads(96, 1), vec(96, 1, 1));
  EXPECT_EQ(ctxThreads(128, 1), vec(128, 1, 1));

  EXPECT_EQ(ctxThreads(1, 2), vec(1, 1, 1));
  EXPECT_EQ(ctxThreads(2, 2), vec(2, 1, 1));
  EXPECT_EQ(ctxThreads(3, 2), vec(3, 1, 1));
  EXPECT_EQ(ctxThreads(4, 2), vec(2, 2, 1));
  EXPECT_EQ(ctxThreads(5, 2), vec(2, 2, 1));
  EXPECT_EQ(ctxThreads(6, 2), vec(3, 2, 1));
  EXPECT_EQ(ctxThreads(7, 2), vec(3, 2, 1));
  EXPECT_EQ(ctxThreads(8, 2), vec(4, 2, 1));
  EXPECT_EQ(ctxThreads(12, 2), vec(4, 3, 1));
  EXPECT_EQ(ctxThreads(16, 2), vec(4, 4, 1));
  EXPECT_EQ(ctxThreads(20, 2), vec(5, 4, 1));
  EXPECT_EQ(ctxThreads(24, 2), vec(6, 4, 1));
  EXPECT_EQ(ctxThreads(32, 2), vec(8, 4, 1));
  EXPECT_EQ(ctxThreads(36, 2), vec(6, 6, 1));
  EXPECT_EQ(ctxThreads(40, 2), vec(8, 5, 1));
  EXPECT_EQ(ctxThreads(48, 2), vec(8, 6, 1));
  EXPECT_EQ(ctxThreads(64, 2), vec(8, 8, 1));
  EXPECT_EQ(ctxThreads(72, 2), vec(9, 8, 1));
  EXPECT_EQ(ctxThreads(80, 2), vec(10, 8, 1));
  EXPECT_EQ(ctxThreads(96, 2), vec(12, 8, 1));
  EXPECT_EQ(ctxThreads(128, 2), vec(16, 8, 1));

  EXPECT_EQ(ctxThreads(1, 3), vec(1, 1, 1));
  EXPECT_EQ(ctxThreads(2, 3), vec(2, 1, 1));
  EXPECT_EQ(ctxThreads(3, 3), vec(3, 1, 1));
  EXPECT_EQ(ctxThreads(4, 3), vec(2, 2, 1));
  EXPECT_EQ(ctxThreads(5, 3), vec(2, 2, 1));
  EXPECT_EQ(ctxThreads(6, 3), vec(3, 2, 1));
  EXPECT_EQ(ctxThreads(7, 3), vec(3, 2, 1));
  EXPECT_EQ(ctxThreads(8, 3), vec(2, 2, 2));
  EXPECT_EQ(ctxThreads(12, 3), vec(3, 2, 2));
  EXPECT_EQ(ctxThreads(16, 3), vec(4, 2, 2));
  EXPECT_EQ(ctxThreads(20, 3), vec(3, 3, 2));
  EXPECT_EQ(ctxThreads(24, 3), vec(4, 3, 2));
  EXPECT_EQ(ctxThreads(32, 3), vec(4, 4, 2));
  EXPECT_EQ(ctxThreads(36, 3), vec(4, 3, 3));
  EXPECT_EQ(ctxThreads(40, 3), vec(5, 4, 2));
  EXPECT_EQ(ctxThreads(48, 3), vec(4, 4, 3));
  EXPECT_EQ(ctxThreads(64, 3), vec(4, 4, 4));
  EXPECT_EQ(ctxThreads(72, 3), vec(6, 4, 3));
  EXPECT_EQ(ctxThreads(80, 3), vec(5, 4, 4));
  EXPECT_EQ(ctxThreads(96, 3), vec(6, 4, 4));
  EXPECT_EQ(ctxThreads(128, 3), vec(8, 4, 4));
}

TEST(KitrtUtils, ctxThreads) {
  auto ctxNumThreads = [](uint64_t totalThreads, uint32_t depth) -> uint64_t {
    LoopNestContext *ctx = nestBegin(totalThreads, depth);
    uint64_t threads = ctx->usedThreads;
    nestEnd(ctx);

    return threads;
  };

  EXPECT_EQ(ctxNumThreads(1, 1), 1U);
  EXPECT_EQ(ctxNumThreads(2, 1), 2U);
  EXPECT_EQ(ctxNumThreads(3, 1), 3U);
  EXPECT_EQ(ctxNumThreads(4, 1), 4U);
  EXPECT_EQ(ctxNumThreads(5, 1), 5U);
  EXPECT_EQ(ctxNumThreads(6, 1), 6U);
  EXPECT_EQ(ctxNumThreads(7, 1), 7U);
  EXPECT_EQ(ctxNumThreads(8, 1), 8U);
  EXPECT_EQ(ctxNumThreads(12, 1), 12U);
  EXPECT_EQ(ctxNumThreads(16, 1), 16U);
  EXPECT_EQ(ctxNumThreads(20, 1), 20U);
  EXPECT_EQ(ctxNumThreads(24, 1), 24U);
  EXPECT_EQ(ctxNumThreads(32, 1), 32U);
  EXPECT_EQ(ctxNumThreads(36, 1), 36U);
  EXPECT_EQ(ctxNumThreads(40, 1), 40U);
  EXPECT_EQ(ctxNumThreads(48, 1), 48U);
  EXPECT_EQ(ctxNumThreads(64, 1), 64U);
  EXPECT_EQ(ctxNumThreads(72, 1), 72U);
  EXPECT_EQ(ctxNumThreads(80, 1), 80U);
  EXPECT_EQ(ctxNumThreads(96, 1), 96U);
  EXPECT_EQ(ctxNumThreads(128, 1), 128U);

  EXPECT_EQ(ctxNumThreads(1, 2), 1U);
  EXPECT_EQ(ctxNumThreads(2, 2), 2U);
  EXPECT_EQ(ctxNumThreads(3, 2), 3U);
  EXPECT_EQ(ctxNumThreads(4, 2), 4U);
  EXPECT_EQ(ctxNumThreads(5, 2), 4U);
  EXPECT_EQ(ctxNumThreads(6, 2), 6U);
  EXPECT_EQ(ctxNumThreads(7, 2), 6U);
  EXPECT_EQ(ctxNumThreads(8, 2), 8U);
  EXPECT_EQ(ctxNumThreads(12, 2), 12U);
  EXPECT_EQ(ctxNumThreads(16, 2), 16U);
  EXPECT_EQ(ctxNumThreads(20, 2), 20U);
  EXPECT_EQ(ctxNumThreads(24, 2), 24U);
  EXPECT_EQ(ctxNumThreads(32, 2), 32U);
  EXPECT_EQ(ctxNumThreads(36, 2), 36U);
  EXPECT_EQ(ctxNumThreads(40, 2), 40U);
  EXPECT_EQ(ctxNumThreads(48, 2), 48U);
  EXPECT_EQ(ctxNumThreads(64, 2), 64U);
  EXPECT_EQ(ctxNumThreads(72, 2), 72U);
  EXPECT_EQ(ctxNumThreads(80, 2), 80U);
  EXPECT_EQ(ctxNumThreads(96, 2), 96U);
  EXPECT_EQ(ctxNumThreads(128, 2), 128U);

  EXPECT_EQ(ctxNumThreads(1, 3), 1U);
  EXPECT_EQ(ctxNumThreads(2, 3), 2U);
  EXPECT_EQ(ctxNumThreads(3, 3), 3U);
  EXPECT_EQ(ctxNumThreads(4, 3), 4U);
  EXPECT_EQ(ctxNumThreads(5, 3), 4U);
  EXPECT_EQ(ctxNumThreads(6, 3), 6U);
  EXPECT_EQ(ctxNumThreads(7, 3), 6U);
  EXPECT_EQ(ctxNumThreads(8, 3), 8U);
  EXPECT_EQ(ctxNumThreads(12, 3), 12U);
  EXPECT_EQ(ctxNumThreads(16, 3), 16U);
  EXPECT_EQ(ctxNumThreads(20, 3), 18U);
  EXPECT_EQ(ctxNumThreads(24, 3), 24U);
  EXPECT_EQ(ctxNumThreads(32, 3), 32U);
  EXPECT_EQ(ctxNumThreads(36, 3), 36U);
  EXPECT_EQ(ctxNumThreads(40, 3), 40U);
  EXPECT_EQ(ctxNumThreads(48, 3), 48U);
  EXPECT_EQ(ctxNumThreads(64, 3), 64U);
  EXPECT_EQ(ctxNumThreads(72, 3), 72U);
  EXPECT_EQ(ctxNumThreads(80, 3), 80U);
  EXPECT_EQ(ctxNumThreads(96, 3), 96U);
  EXPECT_EQ(ctxNumThreads(128, 3), 128U);
}

TEST(KitrtUtils, ctxBounds) {
  auto ctxBounds = [](uint64_t totalThreads, uint32_t depth, uint64_t tid,
                      uint64_t beg, uint64_t end,
                      uint32_t level) -> LoopBounds {
    LoopNestContext *ctx = nestBegin(totalThreads, depth);
    LoopBounds bounds = nestBounds(*ctx, tid, beg, end, level);
    nestEnd(ctx);

    return bounds;
  };

  // forall (i : [0, 12])
  EXPECT_EQ(ctxBounds(1, 1, 0, 0, 12, 1), LoopBounds(0, 12));

  // forall (i : [0, 19])
  //   forall (j : [0, 26])
  EXPECT_EQ(ctxBounds(12, 2, 0, 0, 19, 1), LoopBounds(0, 5));
  EXPECT_EQ(ctxBounds(12, 2, 1, 0, 19, 1), LoopBounds(0, 5));
  EXPECT_EQ(ctxBounds(12, 2, 2, 0, 19, 1), LoopBounds(0, 5));
  EXPECT_EQ(ctxBounds(12, 2, 3, 0, 19, 1), LoopBounds(5, 10));
  EXPECT_EQ(ctxBounds(12, 2, 4, 0, 19, 1), LoopBounds(5, 10));
  EXPECT_EQ(ctxBounds(12, 2, 5, 0, 19, 1), LoopBounds(5, 10));
  EXPECT_EQ(ctxBounds(12, 2, 6, 0, 19, 1), LoopBounds(10, 15));
  EXPECT_EQ(ctxBounds(12, 2, 7, 0, 19, 1), LoopBounds(10, 15));
  EXPECT_EQ(ctxBounds(12, 2, 8, 0, 19, 1), LoopBounds(10, 15));
  EXPECT_EQ(ctxBounds(12, 2, 9, 0, 19, 1), LoopBounds(15, 19));
  EXPECT_EQ(ctxBounds(12, 2, 10, 0, 19, 1), LoopBounds(15, 19));
  EXPECT_EQ(ctxBounds(12, 2, 11, 0, 19, 1), LoopBounds(15, 19));

  EXPECT_EQ(ctxBounds(12, 2, 0, 0, 26, 2), LoopBounds(0, 9));
  EXPECT_EQ(ctxBounds(12, 2, 1, 0, 26, 2), LoopBounds(9, 18));
  EXPECT_EQ(ctxBounds(12, 2, 2, 0, 26, 2), LoopBounds(18, 26));
  EXPECT_EQ(ctxBounds(12, 2, 3, 0, 26, 2), LoopBounds(0, 9));
  EXPECT_EQ(ctxBounds(12, 2, 4, 0, 26, 2), LoopBounds(9, 18));
  EXPECT_EQ(ctxBounds(12, 2, 5, 0, 26, 2), LoopBounds(18, 26));
  EXPECT_EQ(ctxBounds(12, 2, 6, 0, 26, 2), LoopBounds(0, 9));
  EXPECT_EQ(ctxBounds(12, 2, 7, 0, 26, 2), LoopBounds(9, 18));
  EXPECT_EQ(ctxBounds(12, 2, 8, 0, 26, 2), LoopBounds(18, 26));
  EXPECT_EQ(ctxBounds(12, 2, 9, 0, 26, 2), LoopBounds(0, 9));
  EXPECT_EQ(ctxBounds(12, 2, 10, 0, 26, 2), LoopBounds(9, 18));
  EXPECT_EQ(ctxBounds(12, 2, 11, 0, 26, 2), LoopBounds(18, 26));

  // forall (i : [0, 7])
  //   forall (j : [0, 10])
  //     forall (k : [0, 20])
  EXPECT_EQ(ctxBounds(24, 3, 0, 0, 7, 1), LoopBounds(0, 2));
  EXPECT_EQ(ctxBounds(24, 3, 1, 0, 7, 1), LoopBounds(0, 2));
  EXPECT_EQ(ctxBounds(24, 3, 2, 0, 7, 1), LoopBounds(0, 2));
  EXPECT_EQ(ctxBounds(24, 3, 3, 0, 7, 1), LoopBounds(0, 2));
  EXPECT_EQ(ctxBounds(24, 3, 4, 0, 7, 1), LoopBounds(0, 2));
  EXPECT_EQ(ctxBounds(24, 3, 5, 0, 7, 1), LoopBounds(0, 2));
  EXPECT_EQ(ctxBounds(24, 3, 6, 0, 7, 1), LoopBounds(2, 4));
  EXPECT_EQ(ctxBounds(24, 3, 7, 0, 7, 1), LoopBounds(2, 4));
  EXPECT_EQ(ctxBounds(24, 3, 8, 0, 7, 1), LoopBounds(2, 4));
  EXPECT_EQ(ctxBounds(24, 3, 9, 0, 7, 1), LoopBounds(2, 4));
  EXPECT_EQ(ctxBounds(24, 3, 10, 0, 7, 1), LoopBounds(2, 4));
  EXPECT_EQ(ctxBounds(24, 3, 11, 0, 7, 1), LoopBounds(2, 4));
  EXPECT_EQ(ctxBounds(24, 3, 12, 0, 7, 1), LoopBounds(4, 6));
  EXPECT_EQ(ctxBounds(24, 3, 13, 0, 7, 1), LoopBounds(4, 6));
  EXPECT_EQ(ctxBounds(24, 3, 14, 0, 7, 1), LoopBounds(4, 6));
  EXPECT_EQ(ctxBounds(24, 3, 15, 0, 7, 1), LoopBounds(4, 6));
  EXPECT_EQ(ctxBounds(24, 3, 16, 0, 7, 1), LoopBounds(4, 6));
  EXPECT_EQ(ctxBounds(24, 3, 17, 0, 7, 1), LoopBounds(4, 6));
  EXPECT_EQ(ctxBounds(24, 3, 18, 0, 7, 1), LoopBounds(6, 7));
  EXPECT_EQ(ctxBounds(24, 3, 19, 0, 7, 1), LoopBounds(6, 7));
  EXPECT_EQ(ctxBounds(24, 3, 20, 0, 7, 1), LoopBounds(6, 7));
  EXPECT_EQ(ctxBounds(24, 3, 21, 0, 7, 1), LoopBounds(6, 7));
  EXPECT_EQ(ctxBounds(24, 3, 22, 0, 7, 1), LoopBounds(6, 7));
  EXPECT_EQ(ctxBounds(24, 3, 23, 0, 7, 1), LoopBounds(6, 7));

  EXPECT_EQ(ctxBounds(24, 3, 0, 0, 10, 2), LoopBounds(0, 4));
  EXPECT_EQ(ctxBounds(24, 3, 1, 0, 10, 2), LoopBounds(0, 4));
  EXPECT_EQ(ctxBounds(24, 3, 2, 0, 10, 2), LoopBounds(4, 8));
  EXPECT_EQ(ctxBounds(24, 3, 3, 0, 10, 2), LoopBounds(4, 8));
  EXPECT_EQ(ctxBounds(24, 3, 4, 0, 10, 2), LoopBounds(8, 10));
  EXPECT_EQ(ctxBounds(24, 3, 5, 0, 10, 2), LoopBounds(8, 10));
  EXPECT_EQ(ctxBounds(24, 3, 6, 0, 10, 2), LoopBounds(0, 4));
  EXPECT_EQ(ctxBounds(24, 3, 7, 0, 10, 2), LoopBounds(0, 4));
  EXPECT_EQ(ctxBounds(24, 3, 8, 0, 10, 2), LoopBounds(4, 8));
  EXPECT_EQ(ctxBounds(24, 3, 9, 0, 10, 2), LoopBounds(4, 8));
  EXPECT_EQ(ctxBounds(24, 3, 10, 0, 10, 2), LoopBounds(8, 10));
  EXPECT_EQ(ctxBounds(24, 3, 11, 0, 10, 2), LoopBounds(8, 10));
  EXPECT_EQ(ctxBounds(24, 3, 12, 0, 10, 2), LoopBounds(0, 4));
  EXPECT_EQ(ctxBounds(24, 3, 13, 0, 10, 2), LoopBounds(0, 4));
  EXPECT_EQ(ctxBounds(24, 3, 14, 0, 10, 2), LoopBounds(4, 8));
  EXPECT_EQ(ctxBounds(24, 3, 15, 0, 10, 2), LoopBounds(4, 8));
  EXPECT_EQ(ctxBounds(24, 3, 16, 0, 10, 2), LoopBounds(8, 10));
  EXPECT_EQ(ctxBounds(24, 3, 17, 0, 10, 2), LoopBounds(8, 10));
  EXPECT_EQ(ctxBounds(24, 3, 18, 0, 10, 2), LoopBounds(0, 4));
  EXPECT_EQ(ctxBounds(24, 3, 19, 0, 10, 2), LoopBounds(0, 4));
  EXPECT_EQ(ctxBounds(24, 3, 20, 0, 10, 2), LoopBounds(4, 8));
  EXPECT_EQ(ctxBounds(24, 3, 21, 0, 10, 2), LoopBounds(4, 8));
  EXPECT_EQ(ctxBounds(24, 3, 22, 0, 10, 2), LoopBounds(8, 10));
  EXPECT_EQ(ctxBounds(24, 3, 23, 0, 10, 2), LoopBounds(8, 10));

  EXPECT_EQ(ctxBounds(24, 3, 0, 0, 20, 3), LoopBounds(0, 10));
  EXPECT_EQ(ctxBounds(24, 3, 1, 0, 20, 3), LoopBounds(10, 20));
  EXPECT_EQ(ctxBounds(24, 3, 2, 0, 20, 3), LoopBounds(0, 10));
  EXPECT_EQ(ctxBounds(24, 3, 3, 0, 20, 3), LoopBounds(10, 20));
  EXPECT_EQ(ctxBounds(24, 3, 4, 0, 20, 3), LoopBounds(0, 10));
  EXPECT_EQ(ctxBounds(24, 3, 5, 0, 20, 3), LoopBounds(10, 20));
  EXPECT_EQ(ctxBounds(24, 3, 6, 0, 20, 3), LoopBounds(0, 10));
  EXPECT_EQ(ctxBounds(24, 3, 7, 0, 20, 3), LoopBounds(10, 20));
  EXPECT_EQ(ctxBounds(24, 3, 8, 0, 20, 3), LoopBounds(0, 10));
  EXPECT_EQ(ctxBounds(24, 3, 9, 0, 20, 3), LoopBounds(10, 20));
  EXPECT_EQ(ctxBounds(24, 3, 10, 0, 20, 3), LoopBounds(0, 10));
  EXPECT_EQ(ctxBounds(24, 3, 11, 0, 20, 3), LoopBounds(10, 20));
  EXPECT_EQ(ctxBounds(24, 3, 12, 0, 20, 3), LoopBounds(0, 10));
  EXPECT_EQ(ctxBounds(24, 3, 13, 0, 20, 3), LoopBounds(10, 20));
  EXPECT_EQ(ctxBounds(24, 3, 14, 0, 20, 3), LoopBounds(0, 10));
  EXPECT_EQ(ctxBounds(24, 3, 15, 0, 20, 3), LoopBounds(10, 20));
  EXPECT_EQ(ctxBounds(24, 3, 16, 0, 20, 3), LoopBounds(0, 10));
  EXPECT_EQ(ctxBounds(24, 3, 17, 0, 20, 3), LoopBounds(10, 20));
  EXPECT_EQ(ctxBounds(24, 3, 18, 0, 20, 3), LoopBounds(0, 10));
  EXPECT_EQ(ctxBounds(24, 3, 19, 0, 20, 3), LoopBounds(10, 20));
  EXPECT_EQ(ctxBounds(24, 3, 20, 0, 20, 3), LoopBounds(0, 10));
  EXPECT_EQ(ctxBounds(24, 3, 21, 0, 20, 3), LoopBounds(10, 20));
  EXPECT_EQ(ctxBounds(24, 3, 22, 0, 20, 3), LoopBounds(0, 10));
  EXPECT_EQ(ctxBounds(24, 3, 23, 0, 20, 3), LoopBounds(10, 20));

  // forall (i : [0, 5])
  //   forall (j : [0, 14])
  //     forall (k : [0, 11])
  EXPECT_EQ(ctxBounds(6, 3, 0, 0, 5, 1), LoopBounds(0, 2));
  EXPECT_EQ(ctxBounds(6, 3, 1, 0, 5, 1), LoopBounds(0, 2));
  EXPECT_EQ(ctxBounds(6, 3, 2, 0, 5, 1), LoopBounds(2, 4));
  EXPECT_EQ(ctxBounds(6, 3, 3, 0, 5, 1), LoopBounds(2, 4));
  EXPECT_EQ(ctxBounds(6, 3, 4, 0, 5, 1), LoopBounds(4, 5));
  EXPECT_EQ(ctxBounds(6, 3, 5, 0, 5, 1), LoopBounds(4, 5));

  EXPECT_EQ(ctxBounds(6, 3, 0, 0, 14, 2), LoopBounds(0, 7));
  EXPECT_EQ(ctxBounds(6, 3, 1, 0, 14, 2), LoopBounds(7, 14));
  EXPECT_EQ(ctxBounds(6, 3, 2, 0, 14, 2), LoopBounds(0, 7));
  EXPECT_EQ(ctxBounds(6, 3, 3, 0, 14, 2), LoopBounds(7, 14));
  EXPECT_EQ(ctxBounds(6, 3, 4, 0, 14, 2), LoopBounds(0, 7));
  EXPECT_EQ(ctxBounds(6, 3, 5, 0, 14, 2), LoopBounds(7, 14));

  EXPECT_EQ(ctxBounds(6, 3, 0, 0, 11, 3), LoopBounds(0, 11));
  EXPECT_EQ(ctxBounds(6, 3, 1, 0, 11, 3), LoopBounds(0, 11));
  EXPECT_EQ(ctxBounds(6, 3, 2, 0, 11, 3), LoopBounds(0, 11));
  EXPECT_EQ(ctxBounds(6, 3, 3, 0, 11, 3), LoopBounds(0, 11));
  EXPECT_EQ(ctxBounds(6, 3, 4, 0, 11, 3), LoopBounds(0, 11));
  EXPECT_EQ(ctxBounds(6, 3, 5, 0, 11, 3), LoopBounds(0, 11));
}

} // namespace

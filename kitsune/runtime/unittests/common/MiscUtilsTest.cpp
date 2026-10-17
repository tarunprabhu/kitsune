//===- MiscUtilsTest.cpp - Tests for miscellaneous utilities --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common/env.h"
#include "common/utils.h"

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

TEST(KitrtUtils, getNumThreadsForNestLevel_1) {
  EXPECT_EQ(getNumThreadsForNestLevel(1, 1, 1), 1U);
  EXPECT_EQ(getNumThreadsForNestLevel(2, 1, 1), 2U);
  EXPECT_EQ(getNumThreadsForNestLevel(3, 1, 1), 3U);
  EXPECT_EQ(getNumThreadsForNestLevel(4, 1, 1), 4U);
  EXPECT_EQ(getNumThreadsForNestLevel(5, 1, 1), 5U);
  EXPECT_EQ(getNumThreadsForNestLevel(6, 1, 1), 6U);
  EXPECT_EQ(getNumThreadsForNestLevel(7, 1, 1), 7U);
  EXPECT_EQ(getNumThreadsForNestLevel(8, 1, 1), 8U);
  EXPECT_EQ(getNumThreadsForNestLevel(12, 1, 1), 12U);
  EXPECT_EQ(getNumThreadsForNestLevel(16, 1, 1), 16U);
  EXPECT_EQ(getNumThreadsForNestLevel(20, 1, 1), 20U);
  EXPECT_EQ(getNumThreadsForNestLevel(24, 1, 1), 24U);
  EXPECT_EQ(getNumThreadsForNestLevel(32, 1, 1), 32U);
  EXPECT_EQ(getNumThreadsForNestLevel(36, 1, 1), 36U);
  EXPECT_EQ(getNumThreadsForNestLevel(40, 1, 1), 40U);
  EXPECT_EQ(getNumThreadsForNestLevel(48, 1, 1), 48U);
  EXPECT_EQ(getNumThreadsForNestLevel(64, 1, 1), 64U);
  EXPECT_EQ(getNumThreadsForNestLevel(72, 1, 1), 72U);
  EXPECT_EQ(getNumThreadsForNestLevel(80, 1, 1), 80U);
  EXPECT_EQ(getNumThreadsForNestLevel(96, 1, 1), 96U);
  EXPECT_EQ(getNumThreadsForNestLevel(128, 1, 1), 128U);
}

TEST(KitrtUtils, getNumThreadsForNestLevel_2) {
  EXPECT_EQ(getNumThreadsForNestLevel(1, 2, 1), 1U);    // 1 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(2, 2, 1), 2U);    // 2 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(3, 2, 1), 3U);    // 3 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(4, 2, 1), 2U);    // 2 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(5, 2, 1), 2U);    // 2 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(6, 2, 1), 3U);    // 3 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(7, 2, 1), 3U);    // 3 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(8, 2, 1), 4U);    // 4 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(12, 2, 1), 4U);   // 4 x 3
  EXPECT_EQ(getNumThreadsForNestLevel(16, 2, 1), 4U);   // 4 x 4
  EXPECT_EQ(getNumThreadsForNestLevel(20, 2, 1), 5U);   // 5 x 4
  EXPECT_EQ(getNumThreadsForNestLevel(24, 2, 1), 6U);   // 6 x 4
  EXPECT_EQ(getNumThreadsForNestLevel(32, 2, 1), 8U);   // 8 x 4
  EXPECT_EQ(getNumThreadsForNestLevel(36, 2, 1), 6U);   // 6 x 6
  EXPECT_EQ(getNumThreadsForNestLevel(40, 2, 1), 8U);   // 8 x 5
  EXPECT_EQ(getNumThreadsForNestLevel(48, 2, 1), 8U);   // 8 x 6
  EXPECT_EQ(getNumThreadsForNestLevel(64, 2, 1), 8U);   // 8 x 8
  EXPECT_EQ(getNumThreadsForNestLevel(72, 2, 1), 9U);   // 9 x 8
  EXPECT_EQ(getNumThreadsForNestLevel(80, 2, 1), 10U);  // 10 x 8
  EXPECT_EQ(getNumThreadsForNestLevel(96, 2, 1), 12U);  // 12 x 8
  EXPECT_EQ(getNumThreadsForNestLevel(128, 2, 1), 16U); // 16 x 8

  EXPECT_EQ(getNumThreadsForNestLevel(1, 2, 2), 1U);   // 1 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(2, 2, 2), 1U);   // 2 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(3, 2, 2), 1U);   // 3 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(4, 2, 2), 2U);   // 2 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(5, 2, 2), 2U);   // 2 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(6, 2, 2), 2U);   // 3 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(7, 2, 2), 2U);   // 3 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(8, 2, 2), 2U);   // 4 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(12, 2, 2), 3U);  // 4 x 3
  EXPECT_EQ(getNumThreadsForNestLevel(16, 2, 2), 4U);  // 4 x 4
  EXPECT_EQ(getNumThreadsForNestLevel(20, 2, 2), 4U);  // 5 x 4
  EXPECT_EQ(getNumThreadsForNestLevel(24, 2, 2), 4U);  // 6 x 4
  EXPECT_EQ(getNumThreadsForNestLevel(32, 2, 2), 4U);  // 8 x 4
  EXPECT_EQ(getNumThreadsForNestLevel(36, 2, 2), 6U);  // 6 x 6
  EXPECT_EQ(getNumThreadsForNestLevel(40, 2, 2), 5U);  // 8 x 5
  EXPECT_EQ(getNumThreadsForNestLevel(48, 2, 2), 6U);  // 8 x 6
  EXPECT_EQ(getNumThreadsForNestLevel(64, 2, 2), 8U);  // 8 x 8
  EXPECT_EQ(getNumThreadsForNestLevel(72, 2, 2), 8U);  // 9 x 8
  EXPECT_EQ(getNumThreadsForNestLevel(80, 2, 2), 8U);  // 10 x 8
  EXPECT_EQ(getNumThreadsForNestLevel(96, 2, 2), 8U);  // 12 x 8
  EXPECT_EQ(getNumThreadsForNestLevel(128, 2, 2), 8U); // 16 x 8
}

TEST(KitrtUtils, getNumThreadsForNestLevel_3) {
  EXPECT_EQ(getNumThreadsForNestLevel(1, 3, 1), 1U);   // 1 x 1 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(2, 3, 1), 2U);   // 2 x 1 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(3, 3, 1), 3U);   // 3 x 1 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(4, 3, 1), 2U);   // 2 x 2 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(5, 3, 1), 2U);   // 2 x 2 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(6, 3, 1), 3U);   // 3 x 2 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(7, 3, 1), 3U);   // 3 x 2 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(8, 3, 1), 2U);   // 2 x 2 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(12, 3, 1), 3U);  // 3 x 2 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(16, 3, 1), 4U);  // 4 x 2 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(20, 3, 1), 3U);  // 3 x 3 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(24, 3, 1), 4U);  // 4 x 3 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(32, 3, 1), 4U);  // 4 x 4 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(36, 3, 1), 4U);  // 4 x 3 x 3
  EXPECT_EQ(getNumThreadsForNestLevel(40, 3, 1), 5U);  // 5 x 4 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(48, 3, 1), 4U);  // 4 x 4 x 3
  EXPECT_EQ(getNumThreadsForNestLevel(64, 3, 1), 4U);  // 4 x 4 x 4
  EXPECT_EQ(getNumThreadsForNestLevel(72, 3, 1), 6U);  // 6 x 4 x 3
  EXPECT_EQ(getNumThreadsForNestLevel(80, 3, 1), 5U);  // 5 x 4 x 4
  EXPECT_EQ(getNumThreadsForNestLevel(96, 3, 1), 6U);  // 6 x 4 x 4
  EXPECT_EQ(getNumThreadsForNestLevel(128, 3, 1), 8U); // 8 x 4 x 4

  EXPECT_EQ(getNumThreadsForNestLevel(1, 3, 2), 1U);   // 1 x 1 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(2, 3, 2), 1U);   // 2 x 1 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(3, 3, 2), 1U);   // 3 x 1 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(4, 3, 2), 2U);   // 2 x 2 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(5, 3, 2), 2U);   // 2 x 2 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(6, 3, 2), 2U);   // 3 x 2 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(7, 3, 2), 2U);   // 3 x 2 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(8, 3, 2), 2U);   // 2 x 2 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(12, 3, 2), 2U);  // 3 x 2 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(16, 3, 2), 2U);  // 4 x 2 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(20, 3, 2), 3U);  // 3 x 3 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(24, 3, 2), 3U);  // 4 x 3 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(32, 3, 2), 4U);  // 4 x 4 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(36, 3, 2), 3U);  // 4 x 3 x 3
  EXPECT_EQ(getNumThreadsForNestLevel(40, 3, 2), 4U);  // 5 x 4 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(48, 3, 2), 4U);  // 4 x 4 x 3
  EXPECT_EQ(getNumThreadsForNestLevel(64, 3, 2), 4U);  // 4 x 4 x 4
  EXPECT_EQ(getNumThreadsForNestLevel(72, 3, 2), 4U);  // 6 x 4 x 3
  EXPECT_EQ(getNumThreadsForNestLevel(80, 3, 2), 4U);  // 5 x 4 x 4
  EXPECT_EQ(getNumThreadsForNestLevel(96, 3, 2), 4U);  // 6 x 4 x 4
  EXPECT_EQ(getNumThreadsForNestLevel(128, 3, 2), 4U); // 8 x 4 x 4

  EXPECT_EQ(getNumThreadsForNestLevel(1, 3, 3), 1U);   // 1 x 1 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(2, 3, 3), 1U);   // 2 x 1 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(3, 3, 3), 1U);   // 3 x 1 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(4, 3, 3), 1U);   // 2 x 2 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(5, 3, 3), 1U);   // 2 x 2 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(6, 3, 3), 1U);   // 3 x 2 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(7, 3, 3), 1U);   // 3 x 2 x 1
  EXPECT_EQ(getNumThreadsForNestLevel(8, 3, 3), 2U);   // 2 x 2 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(12, 3, 3), 2U);  // 3 x 2 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(16, 3, 3), 2U);  // 4 x 2 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(20, 3, 3), 2U);  // 3 x 3 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(24, 3, 3), 2U);  // 4 x 3 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(32, 3, 3), 2U);  // 4 x 4 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(36, 3, 3), 3U);  // 4 x 3 x 3
  EXPECT_EQ(getNumThreadsForNestLevel(40, 3, 3), 2U);  // 5 x 4 x 2
  EXPECT_EQ(getNumThreadsForNestLevel(48, 3, 3), 3U);  // 4 x 4 x 3
  EXPECT_EQ(getNumThreadsForNestLevel(64, 3, 3), 4U);  // 4 x 4 x 4
  EXPECT_EQ(getNumThreadsForNestLevel(72, 3, 3), 3U);  // 6 x 4 x 3
  EXPECT_EQ(getNumThreadsForNestLevel(80, 3, 3), 4U);  // 5 x 4 x 4
  EXPECT_EQ(getNumThreadsForNestLevel(96, 3, 3), 4U);  // 6 x 4 x 4
  EXPECT_EQ(getNumThreadsForNestLevel(128, 3, 3), 4U); // 8 x 4 x 4
}

} // namespace

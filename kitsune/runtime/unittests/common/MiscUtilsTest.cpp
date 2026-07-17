//===- utilsTest.cpp - Tests for miscellaneous utilities ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common/env.h"
#include "kitrt.h"

#include "gtest/gtest.h"

using namespace kitrt;

namespace {

TEST(KitrtUtils, num_cpus) {
  // We can only check that we always return at least 1. There isn't any
  // advantage to trying to check that the "correct" value for the number of
  // CPU's is returned because the implementation simply calls
  // std::thread::hardware_concurrency.
  EXPECT_GE(__kitrt_num_cpus(), 1U);
}

TEST(KitrtUtils, num_threads) {
  uint32_t cpus = __kitrt_num_cpus();
  EXPECT_EQ(__kitrt_num_threads(nullptr), cpus);

  envSet("KIT_NUM_THREADS", 41U);
  envSet("ALTERNATIVE", 97U);
  EXPECT_EQ(__kitrt_num_threads(nullptr), 41U);
  EXPECT_EQ(__kitrt_num_threads("ALTERNATIVE"), 41U);

  envUnset("KIT_NUM_THREADS");
  EXPECT_EQ(__kitrt_num_threads(nullptr), cpus);
  EXPECT_EQ(__kitrt_num_threads("ALTERNATIVE"), 97U);

  envUnset("ALTERNATIVE");
  EXPECT_EQ(__kitrt_num_threads("KIT_NUM_THREADS"), cpus);
  EXPECT_EQ(__kitrt_num_threads("ALTERNATIVE"), cpus);

  envSet("KIT_NUM_THREADS", "forty-one");
  envSet("ALTERNATIVE", "ninety-seven");
  EXPECT_EQ(__kitrt_num_threads(nullptr), cpus);
  EXPECT_EQ(__kitrt_num_threads("ALTERNATIVE"), cpus);

  envSet("KIT_NUM_THREADS", 0U);
  envSet("ALTERNATIVE", 11U);
  EXPECT_EQ(__kitrt_num_threads(nullptr), cpus);
  EXPECT_EQ(__kitrt_num_threads("ALTERNATIVE"), 11U);

  envSet("KIT_NUM_THREADS", "0");
  envSet("ALTERNATIVE", 0U);
  EXPECT_EQ(__kitrt_num_threads("ALTERNATIVE"), cpus);

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

} // namespace

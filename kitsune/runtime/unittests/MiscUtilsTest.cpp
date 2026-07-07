//===- utilsTest.cpp - Tests for miscellaneous utilities ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitrt.h"

#include "gtest/gtest.h"

namespace {

TEST(KitrtUtils, num_cpus) { EXPECT_GE(__kitrt_num_cpus(), 1U); }

TEST(KitrtUtils, num_threads) {
  unsigned cpus = __kitrt_num_cpus();
  EXPECT_EQ(__kitrt_num_threads(nullptr), cpus);

  __kitrt_env_set("KIT_NUM_THREADS", 41U);
  __kitrt_env_set("ALTERNATIVE", 97U);
  EXPECT_EQ(__kitrt_num_threads(nullptr), 41U);
  EXPECT_EQ(__kitrt_num_threads("ALTERNATIVE"), 41U);

  __kitrt_env_unset("KIT_NUM_THREADS");
  EXPECT_EQ(__kitrt_num_threads(nullptr), cpus);
  EXPECT_EQ(__kitrt_num_threads("ALTERNATIVE"), 97U);

  __kitrt_env_unset("ALTERNATIVE");
  EXPECT_EQ(__kitrt_num_threads("KIT_NUM_THREADS"), cpus);
  EXPECT_EQ(__kitrt_num_threads("ALTERNATIVE"), cpus);

  __kitrt_env_set("KIT_NUM_THREADS", "forty-one");
  __kitrt_env_set("ALTERNATIVE", "ninety-seven");
  EXPECT_EQ(__kitrt_num_threads(nullptr), cpus);
  EXPECT_EQ(__kitrt_num_threads("ALTERNATIVE"), cpus);

  __kitrt_env_unset("KIT_NUM_THREADS");
  __kitrt_env_unset("ALTERNATIVE");
}

TEST(KitrtUtils, nearestPowerOf2LE) {
  auto pow2 = [](unsigned exp) -> unsigned { return 1U << (exp - 1); };

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

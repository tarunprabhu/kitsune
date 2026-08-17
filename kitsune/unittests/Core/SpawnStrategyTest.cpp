//===- SpawnStrategyTest.cpp - Tests for the core TTID enum
//--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/SpawnStrategy.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(SpawnStrategy, toString) {
  EXPECT_EQ(toString(TapirSpawnStrategy::Sequential), "Sequential");
  EXPECT_EQ(toString(TapirSpawnStrategy::DivideAndConquer),
            "Divide and conquer");
  EXPECT_EQ(toString(TapirSpawnStrategy::GPU), "GPU");
  EXPECT_EQ(toString(TapirSpawnStrategy::Basic), "Basic");
}

TEST(KitSpawnStrategy, toInt) {
  EXPECT_EQ(int(TapirSpawnStrategy::Sequential), 1);
  EXPECT_EQ(int(TapirSpawnStrategy::DivideAndConquer), 2);
  EXPECT_EQ(int(TapirSpawnStrategy::GPU), 3);
  EXPECT_EQ(int(TapirSpawnStrategy::Basic), 4);
}

TEST(KitSpawnStrategy, fromInt) {
  // 0 is not ever intended to be a valid value, so check that this is not
  // accidentally changed.
  EXPECT_EQ(fromInt<TapirSpawnStrategy>(0), std::nullopt);
  EXPECT_EQ(fromInt<TapirSpawnStrategy>(5), std::nullopt);

  EXPECT_EQ(fromInt<TapirSpawnStrategy>(1), TapirSpawnStrategy::Sequential);
  EXPECT_EQ(fromInt<TapirSpawnStrategy>(2),
            TapirSpawnStrategy::DivideAndConquer);
  EXPECT_EQ(fromInt<TapirSpawnStrategy>(3), TapirSpawnStrategy::GPU);
  EXPECT_EQ(fromInt<TapirSpawnStrategy>(4), TapirSpawnStrategy::Basic);
}

TEST(KitSpawnStrategy, fromString) {
  EXPECT_EQ(fromString<TapirSpawnStrategy>(""), std::nullopt);
  EXPECT_EQ(fromString<TapirSpawnStrategy>("DAC"), std::nullopt);
  EXPECT_EQ(fromString<TapirSpawnStrategy>("GPU"), std::nullopt);

  EXPECT_EQ(fromString<TapirSpawnStrategy>("seq"),
            TapirSpawnStrategy::Sequential);
  EXPECT_EQ(fromString<TapirSpawnStrategy>("dac"),
            TapirSpawnStrategy::DivideAndConquer);
  EXPECT_EQ(fromString<TapirSpawnStrategy>("gpu"), TapirSpawnStrategy::GPU);
  EXPECT_EQ(fromString<TapirSpawnStrategy>("basic"), TapirSpawnStrategy::Basic);
}

TEST(KitSpawnStrategy, defawlt) {
  // The default spawn strategy is sequential because that does not require
  // outlining, and is only ever used with the serial tapir target.
  EXPECT_EQ(defaultTapirSpawnStrategy, TapirSpawnStrategy::Sequential);
}

} // namespace

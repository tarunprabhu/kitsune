//===- TapirCoreTest.cpp - Tests for the core Tapir types and enums -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/Tapir.h"

#include "gtest/gtest.h"

using namespace llvm;

TEST(TapirCore, tapirTargetValues) {
  // The integer values of the tapir targets should not be changed unless
  // absolutely necessary, so check them here.
  EXPECT_EQ(int(TTID::Nolo), 0x0);
  EXPECT_EQ(int(TTID::Serial), 0x1);
  EXPECT_EQ(int(TTID::Cuda), 0x2);
  EXPECT_EQ(int(TTID::Hip), 0x4);
  EXPECT_EQ(int(TTID::OpenCilk), 0x8);
  // EXPECT_EQ(int(TTID::GPUABI), 0x10);
  EXPECT_EQ(int(TTID::Qthreads), 0x20);
  EXPECT_EQ(int(TTID::Realm), 0x40);
  EXPECT_EQ(int(TTID::Lambda), 0x80);
  EXPECT_EQ(int(TTID::OMPTask), 0x100);
  EXPECT_EQ(int(TTID::OpenMP), 0x200);
  EXPECT_EQ(int(TTID::Pthreads), 0x400);
}

TEST(TapirCore, tapirSpawnStrategyValues) {
  // The integer values of the spawn strategies should not be changed unless
  // absolutely necessary, so check them here.
  EXPECT_EQ(int(TapirSpawnStrategy::Sequential), 1);
  EXPECT_EQ(int(TapirSpawnStrategy::DivideAndConquer), 2);
  EXPECT_EQ(int(TapirSpawnStrategy::GPU), 3);
  EXPECT_EQ(int(TapirSpawnStrategy::Basic), 4);
}

TEST(TapirCore, maybeBoolValues) {
  // The values of the MaybeBool enum need not be the following values, but
  // we do require MaybeBool::Off to be 0. Just in case, check all three.
  EXPECT_EQ(int(MaybeBool::Off), 0);
  EXPECT_EQ(int(MaybeBool::On), 1);
  EXPECT_EQ(int(MaybeBool::Any), 3);
}

TEST(TapirCore, defaults) {
  // We will probably never have a default tapir target.
  EXPECT_EQ(defaultTapirTarget, std::nullopt);

  // The default spawn strategy is sequential because that does not require
  // outlining, and is only ever used with the serial tapir target.
  EXPECT_EQ(defaultTapirSpawnStrategy, TapirSpawnStrategy::Sequential);

  // The default tapir grain size may change if we decide to handle it
  // differently in the various tapir targets. The type may also change to
  // optional.
  EXPECT_EQ(defaultTapirGrainSize, 0U);
}

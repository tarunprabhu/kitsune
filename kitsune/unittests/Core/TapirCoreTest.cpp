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
  EXPECT_EQ(int(TTID::Custom), 0x800);
}

TEST(TapirCore, createTTIDFromInt) {
  EXPECT_EQ(*createTTIDFrom(0x0), TTID::Nolo);
  EXPECT_EQ(*createTTIDFrom(0x1), TTID::Serial);
  EXPECT_EQ(*createTTIDFrom(0x2), TTID::Cuda);
  EXPECT_EQ(*createTTIDFrom(0x4), TTID::Hip);
  EXPECT_EQ(*createTTIDFrom(0x8), TTID::OpenCilk);
  EXPECT_EQ(*createTTIDFrom(0x20), TTID::Qthreads);
  EXPECT_EQ(*createTTIDFrom(0x40), TTID::Realm);
  EXPECT_EQ(*createTTIDFrom(0x80), TTID::Lambda);
  EXPECT_EQ(*createTTIDFrom(0x100), TTID::OMPTask);
  EXPECT_EQ(*createTTIDFrom(0x200), TTID::OpenMP);
  EXPECT_EQ(*createTTIDFrom(0x400), TTID::Pthreads);
  EXPECT_EQ(*createTTIDFrom(0x800), TTID::Custom);
}

TEST(TapirCore, createTTIDFromString) {
  EXPECT_EQ(*createTTIDFrom("nolo"), TTID::Nolo);
  EXPECT_EQ(*createTTIDFrom("serial"), TTID::Serial);
  EXPECT_EQ(*createTTIDFrom("cuda"), TTID::Cuda);
  EXPECT_EQ(*createTTIDFrom("hip"), TTID::Hip);
  EXPECT_EQ(*createTTIDFrom("opencilk"), TTID::OpenCilk);
  EXPECT_EQ(*createTTIDFrom("qthreads"), TTID::Qthreads);
  EXPECT_EQ(*createTTIDFrom("realm"), TTID::Realm);
  EXPECT_EQ(*createTTIDFrom("lambda"), TTID::Lambda);
  EXPECT_EQ(*createTTIDFrom("omptask"), TTID::OMPTask);
  EXPECT_EQ(*createTTIDFrom("openmp"), TTID::OpenMP);
  EXPECT_EQ(*createTTIDFrom("pthreads"), TTID::Pthreads);
  EXPECT_EQ(*createTTIDFrom("custom"), TTID::Custom);
}

TEST(TapirCore, tapirSpawnStrategyValues) {
  // The integer values of the spawn strategies should not be changed unless
  // absolutely necessary, so check them here.
  EXPECT_EQ(int(TapirSpawnStrategy::Sequential), 1);
  EXPECT_EQ(int(TapirSpawnStrategy::DivideAndConquer), 2);
  EXPECT_EQ(int(TapirSpawnStrategy::GPU), 3);
  EXPECT_EQ(int(TapirSpawnStrategy::Basic), 4);
}

TEST(TapirCore, createTapirSpawnStrategyFromInt) {
  EXPECT_EQ(*createTapirSpawnStrategyFrom(1), TapirSpawnStrategy::Sequential);
  EXPECT_EQ(*createTapirSpawnStrategyFrom(2),
            TapirSpawnStrategy::DivideAndConquer);
  EXPECT_EQ(*createTapirSpawnStrategyFrom(3), TapirSpawnStrategy::GPU);
  EXPECT_EQ(*createTapirSpawnStrategyFrom(4), TapirSpawnStrategy::Basic);
}

TEST(TapirCore, createTapirSpawnStrategyFromString) {
  EXPECT_EQ(*createTapirSpawnStrategyFrom("seq"),
            TapirSpawnStrategy::Sequential);
  EXPECT_EQ(*createTapirSpawnStrategyFrom("dac"),
            TapirSpawnStrategy::DivideAndConquer);
  EXPECT_EQ(*createTapirSpawnStrategyFrom("gpu"), TapirSpawnStrategy::GPU);
  EXPECT_EQ(*createTapirSpawnStrategyFrom("basic"), TapirSpawnStrategy::Basic);
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

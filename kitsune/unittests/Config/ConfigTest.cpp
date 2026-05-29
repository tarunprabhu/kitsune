//===- ConfigTest.cpp - Unit tests for configuration utilities ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The purpose of these tests is to ensure that something gets flagged if
// support for a new tapir target is added. Some other tests will iterate over
// the known tapir targets to ensure that testing is as "comprehensive" as
// possible
//
//===----------------------------------------------------------------------===//

#include "kitsune/Config/Config.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitConfigUtils, knownTTs) {
  TTID expected[] = {TTID::Cuda,     TTID::Custom, TTID::Hip,
                     TTID::OpenCilk, TTID::OpenMP, TTID::Pthreads,
                     TTID::Qthreads, TTID::Serial};
  EXPECT_EQ(kitKnownTTs(), ArrayRef<TTID>(expected));
}

TEST(KitConfigUtils, knownGPUTTs) {
  TTID expected[] = {TTID::Cuda, TTID::Hip};
  EXPECT_EQ(kitKnownGPUTTs(), ArrayRef<TTID>(expected));
}

TEST(kitConfigUtils, knownEmbBCTTs) {
  TTID expected[] = {TTID::Cuda, TTID::Hip};
  EXPECT_EQ(kitKnownEmbBCTTs(), ArrayRef<TTID>(expected));
}

TEST(KitConfigUtils, universal) {
  TTID universal[] = {TTID::Custom, TTID::OpenMP, TTID::Pthreads, TTID::Serial};
  EXPECT_EQ(kitUniversalTTs(), ArrayRef<TTID>(universal));

  EXPECT_TRUE(kitCustomEnabled());
  EXPECT_TRUE(kitOpenMPEnabled());
  EXPECT_TRUE(kitPthreadsEnabled());
  EXPECT_TRUE(kitSerialEnabled());
}

TEST(KitConfigUtils, unsupported) {
  // These tapir targets are known to be unsupported, so they should never be
  // show up as enabled.
  EXPECT_FALSE(kitLambdaEnabled());
  EXPECT_FALSE(kitOMPTaskEnabled());
  EXPECT_FALSE(kitRealmEnabled());
}

} // namespace

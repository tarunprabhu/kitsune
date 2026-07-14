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

static bool contains(ArrayRef<TTID> tts, TTID key) {
  for (TTID tt : tts)
    if (tt == key)
      return true;
  return false;
}

TEST(KitConfigUtils, knownTTs) {
  TTID expected[] = {TTID::Cuda,     TTID::Custom, TTID::Hip,
                     TTID::OpenCilk, TTID::OpenMP, TTID::Pthreads,
                     TTID::Qthreads, TTID::Serial};
  EXPECT_EQ(kitKnownTTs(), ArrayRef<TTID>(expected));
}

TEST(KitConfigUtils, knownCPUTTs) {
  TTID expected[] = {TTID::OpenCilk, TTID::OpenMP, TTID::Pthreads,
                     TTID::Qthreads, TTID::Serial};
  EXPECT_EQ(kitKnownCPUTTs(), ArrayRef<TTID>(expected));
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

TEST(KitConfigUtils, enabledCPUTTs) {
  ArrayRef<TTID> tts = kitKnownCPUTTs();
  for (TTID tt : kitEnabledCPUTTs())
    EXPECT_TRUE(contains(tts, tt));
}

TEST(KitConfigUtils, enabledGPUTTs) {
  ArrayRef<TTID> tts = kitKnownGPUTTs();
  for (TTID tt : kitEnabledGPUTTs())
    EXPECT_TRUE(contains(tts, tt));
}

TEST(KitConfigUtils, specialCases) {
  ArrayRef<TTID> known = kitKnownTTs();
  ArrayRef<TTID> cpu = kitKnownCPUTTs();
  ArrayRef<TTID> gpu = kitKnownGPUTTs();

  // TTID::Nolo should never be in the list of known tapir targets since it is
  // a pseudo target that does not generate code.
  EXPECT_FALSE(contains(known, TTID::Nolo));

  // TTID::Custom should never be in the list known CPU or GPU tapir targets
  // since we cannot know what a given custom tapir target can do. But it should
  // be known.
  EXPECT_TRUE(contains(known, TTID::Custom));
  EXPECT_FALSE(contains(cpu, TTID::Custom));
  EXPECT_FALSE(contains(gpu, TTID::Custom));
}

} // namespace

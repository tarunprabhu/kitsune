//===- TTUtilsTest.cpp - Unit tests for tapir target utilities ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/TTUtils.h"
#include "kitsune/Config/Config.h"
#include "llvm/ADT/SetVector.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

static SmallSetVector<TTID, 4> getAsSet(ArrayRef<TTID> tts) {
  return SmallSetVector<TTID, 4>(tts.begin(), tts.end());
}

TEST(KitTTUtils, generatesEmbBC) {
  SmallSetVector<TTID, 4> knownEmbBCTTs = getAsSet(kitKnownEmbBCTTs());
  for (TTID tt : kitKnownTTs())
    EXPECT_EQ(generatesEmbBC(tt), knownEmbBCTTs.contains(tt));
}

TEST(KitTTUtils, isGPUTT) {
  SmallSetVector<TTID, 4> knownGPUTTs = getAsSet(kitKnownGPUTTs());
  for (TTID tt : kitKnownTTs())
    EXPECT_EQ(isGPUTT(tt), knownGPUTTs.contains(tt));
}

TEST(KitTTUtils, isCPUTT) {
  SmallSetVector<TTID, 4> knownCPUTTs = getAsSet(kitKnownCPUTTs());
  for (TTID tt : kitKnownTTs())
    EXPECT_EQ(isCPUTT(tt), knownCPUTTs.contains(tt));
}

TEST(KitTTUtils, getSpawnStrategy) {
  // TTID::Nolo will never be in the known TT's list.
  EXPECT_EQ(getSpawnStrategyFor(TTID::Nolo), TapirSpawnStrategy::Sequential);

  for (TTID tt : kitKnownTTs()) {
    TapirSpawnStrategy strategy = getSpawnStrategyFor(tt);
    switch (tt) {
    case TTID::Serial:
      EXPECT_EQ(strategy, TapirSpawnStrategy::Sequential);
      break;
    case TTID::OpenCilk:
      EXPECT_EQ(strategy, TapirSpawnStrategy::DivideAndConquer);
      break;
    case TTID::Cuda:
    case TTID::Hip:
      EXPECT_EQ(strategy, TapirSpawnStrategy::GPU);
      break;
    case TTID::Custom:
    case TTID::OpenMP:
    case TTID::Pthreads:
    case TTID::Qthreads:
      EXPECT_EQ(strategy, TapirSpawnStrategy::Basic);
      break;
    default:
      FAIL();
      break;
    }
  }
}

TEST(KitTTUtils, isEnabledTT) {
  // It is not clear if there is any advantage in checking that this function
  // works by directly inspecting the variables that this function would itself
  // inspect. So we just check that the special case of TTID::Nolo is handled
  // correctly.
  EXPECT_TRUE(isEnabledTT(TTID::Nolo));
}

} // namespace

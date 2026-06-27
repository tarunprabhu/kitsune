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
  for (TTID tt : kitKnownTTs()) {
    switch (tt) {
    case TTID::Nolo:
    case TTID::Serial:
    case TTID::Cuda:
    case TTID::Hip:
    case TTID::Custom:
      EXPECT_FALSE(isCPUTT(tt));
      continue;
    case TTID::OpenCilk:
    case TTID::OpenMP:
    case TTID::Pthreads:
    case TTID::Qthreads:
      EXPECT_TRUE(isCPUTT(tt));
      continue;
    case TTID::Lambda:
    case TTID::OMPTask:
    case TTID::Realm:
      EXPECT_DEATH(isCPUTT(tt), "isCPUTT: TTID not handled");
      continue;
    }
    FAIL() << "TTID not handled in test switch";
  }
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
  EXPECT_TRUE(isEnabledTT(TTID::Nolo));
  for (TTID tt : kitKnownTTs()) {
    bool enabled = isEnabledTT(tt);
    if (tt == TTID::Cuda)
      EXPECT_EQ(enabled, kitCudaEnabled());
    else if (tt == TTID::Custom)
      EXPECT_EQ(enabled, kitCustomEnabled());
    else if (tt == TTID::Hip)
      EXPECT_EQ(enabled, kitHipEnabled());
    else if (tt == TTID::OpenCilk)
      EXPECT_EQ(enabled, kitOpenCilkEnabled());
    else if (tt == TTID::OpenMP)
      EXPECT_EQ(enabled, kitOpenMPEnabled());
    else if (tt == TTID::Pthreads)
      EXPECT_EQ(enabled, kitPthreadsEnabled());
    else if (tt == TTID::Qthreads)
      EXPECT_EQ(enabled, kitQthreadsEnabled());
    else if (tt == TTID::Serial)
      EXPECT_EQ(enabled, kitSerialEnabled());
    else
      FAIL();
  }
}

} // namespace

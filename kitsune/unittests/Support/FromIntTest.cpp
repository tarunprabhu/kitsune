//===- FromIntTest.cpp - Tests of conversions from integers ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Support/FromInt.h"
#include "kitsune/Core/Tapir.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitFromInt, fromIntTTID) {
  // The integer value of 0x10 was set aside for the GPUABI, but that is being
  // removed from Kitsune. For now, it is unused, so ensure that attempting to
  // convert it always returns std::nullopt.
  EXPECT_EQ(fromInt<TTID>(0x10), std::nullopt);

  // The tapir targets are intentionally intended to have bit patterns that
  // could, potentially be combined. We currently do not support this, so
  // check a few likely patterns.
  EXPECT_EQ(fromInt<TTID>(0x402), std::nullopt);
  EXPECT_EQ(fromInt<TTID>(0x404), std::nullopt);

  EXPECT_EQ(fromInt<TTID>(0x0), TTID::Nolo);
  EXPECT_EQ(fromInt<TTID>(0x1), TTID::Serial);
  EXPECT_EQ(fromInt<TTID>(0x2), TTID::Cuda);
  EXPECT_EQ(fromInt<TTID>(0x4), TTID::Hip);
  EXPECT_EQ(fromInt<TTID>(0x8), TTID::OpenCilk);
  EXPECT_EQ(fromInt<TTID>(0x20), TTID::Qthreads);
  EXPECT_EQ(fromInt<TTID>(0x40), TTID::Realm);
  EXPECT_EQ(fromInt<TTID>(0x80), TTID::Lambda);
  EXPECT_EQ(fromInt<TTID>(0x100), TTID::OMPTask);
  EXPECT_EQ(fromInt<TTID>(0x200), TTID::OpenMP);
  EXPECT_EQ(fromInt<TTID>(0x400), TTID::Pthreads);
  EXPECT_EQ(fromInt<TTID>(0x800), TTID::Custom);
}

TEST(KitFromInt, fromIntTapirSpawnStrategy) {
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

} // namespace

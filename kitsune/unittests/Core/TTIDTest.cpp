//===- KitTT.cpp - Tests for the core TTID enum --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/TTID.h"
#include "kitsune/Config/Config.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitTT, toString) {
  EXPECT_EQ(toString(TTID::Nolo), "nolo");

  // TTID::Nolo will never be returned by kitKnownTTs
  for (TTID tt : kitKnownTTs()) {
    switch (tt) {
    case TTID::Nolo: EXPECT_EQ(toString(tt), "nolo"); break;
    case TTID::Serial: EXPECT_EQ(toString(tt), "serial"); break;
    case TTID::Cuda: EXPECT_EQ(toString(tt), "cuda"); break;
    case TTID::Hip: EXPECT_EQ(toString(tt), "hip"); break;
    case TTID::OpenCilk: EXPECT_EQ(toString(tt), "opencilk"); break;
    case TTID::Qthreads: EXPECT_EQ(toString(tt), "qthreads"); break;
    case TTID::Realm: EXPECT_EQ(toString(tt), "realm"); break;
    case TTID::Lambda: EXPECT_EQ(toString(tt), "lambda"); break;
    case TTID::OMPTask: EXPECT_EQ(toString(tt), "omptask"); break;
    case TTID::OpenMP: EXPECT_EQ(toString(tt), "openmp"); break;
    case TTID::Pthreads: EXPECT_EQ(toString(tt), "pthreads"); break;
    case TTID::Custom: EXPECT_EQ(toString(tt), "custom"); break;
    }
  }
}

TEST(KitTT, toInt) {
  EXPECT_EQ(int(TTID::Nolo), 0x0);

  // TTID::Nolo will never be returned by kitKnownTTs
  for (TTID tt : kitKnownTTs()) {
    switch (tt) {
    case TTID::Nolo: EXPECT_EQ(int(tt), 0x0); break;
    case TTID::Serial: EXPECT_EQ(int(tt), 0x1); break;
    case TTID::Cuda: EXPECT_EQ(int(tt), 0x2); break;
    case TTID::Hip: EXPECT_EQ(int(tt), 0x4); break;
    case TTID::OpenCilk: EXPECT_EQ(int(tt), 0x8); break;
    case TTID::Qthreads: EXPECT_EQ(int(tt), 0x20); break;
    case TTID::Realm: EXPECT_EQ(int(tt), 0x40); break;
    case TTID::Lambda: EXPECT_EQ(int(tt), 0x80); break;
    case TTID::OMPTask: EXPECT_EQ(int(tt), 0x100); break;
    case TTID::OpenMP: EXPECT_EQ(int(tt), 0x200); break;
    case TTID::Pthreads: EXPECT_EQ(int(tt), 0x400); break;
    case TTID::Custom: EXPECT_EQ(int(tt), 0x800); break;
    }
  }
}

TEST(KitTT, fromInt) {
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

TEST(KitTT, fromString) {
  EXPECT_EQ(fromString<TTID>(""), std::nullopt);
  EXPECT_EQ(fromString<TTID>("OpenCilk"), std::nullopt);

  EXPECT_EQ(fromString<TTID>("nolo"), TTID::Nolo);
  EXPECT_EQ(fromString<TTID>("serial"), TTID::Serial);
  EXPECT_EQ(fromString<TTID>("cuda"), TTID::Cuda);
  EXPECT_EQ(fromString<TTID>("hip"), TTID::Hip);
  EXPECT_EQ(fromString<TTID>("opencilk"), TTID::OpenCilk);
  EXPECT_EQ(fromString<TTID>("qthreads"), TTID::Qthreads);
  EXPECT_EQ(fromString<TTID>("realm"), TTID::Realm);
  EXPECT_EQ(fromString<TTID>("lambda"), TTID::Lambda);
  EXPECT_EQ(fromString<TTID>("omptask"), TTID::OMPTask);
  EXPECT_EQ(fromString<TTID>("openmp"), TTID::OpenMP);
  EXPECT_EQ(fromString<TTID>("pthreads"), TTID::Pthreads);
  EXPECT_EQ(fromString<TTID>("custom"), TTID::Custom);
}

TEST(KitTT, defawlt) {
  // We will probably never have a default tapir target.
  EXPECT_EQ(defaultTapirTarget, std::nullopt);
}

} // namespace

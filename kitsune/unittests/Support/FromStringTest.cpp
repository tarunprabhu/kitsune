//===- FromStringTest.cpp - Tests of conversions from strings -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Support/FromString.h"
#include "kitsune/Core/Instrumentation.h"
#include "kitsune/Core/Tapir.h"
#include "kitsune/Support/MaybeBool.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitFromString, fromStringTTID) {
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

TEST(KitFromString, fromStringTapirSpawnStrategy) {
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

TEST(KitFromString, fromStringMaybeBool) {
  EXPECT_EQ(fromString<MaybeBool>(""), std::nullopt);
  EXPECT_EQ(fromString<MaybeBool>("ON"), std::nullopt);
  EXPECT_EQ(fromString<MaybeBool>("OFF"), std::nullopt);

  EXPECT_EQ(fromString<MaybeBool>("off"), MaybeBool::Off);
  EXPECT_EQ(fromString<MaybeBool>("on"), MaybeBool::On);
  EXPECT_EQ(fromString<MaybeBool>("any"), MaybeBool::Any);
}

TEST(KitFromString, fromStringInstrumentKind) {
  EXPECT_EQ(fromString<InstrumentKind>(""), std::nullopt);
  EXPECT_EQ(fromString<InstrumentKind>("PAPI"), std::nullopt);
  EXPECT_EQ(fromString<InstrumentKind>("counter"), std::nullopt);

  EXPECT_EQ(fromString<InstrumentKind>("generic"), InstrumentKind::Generic);
  EXPECT_EQ(fromString<InstrumentKind>("papi"), InstrumentKind::PAPI);
  EXPECT_EQ(fromString<InstrumentKind>("timer"), InstrumentKind::Timer);
}

TEST(KitFromString, fromStringInstrumentUnit) {
  EXPECT_EQ(fromString<InstrumentUnit>(""), std::nullopt);
  EXPECT_EQ(fromString<InstrumentUnit>("Loop"), std::nullopt);
  EXPECT_EQ(fromString<InstrumentUnit>("function"), std::nullopt);

  EXPECT_EQ(fromString<InstrumentUnit>("loop"), InstrumentUnit::Loop);
  EXPECT_EQ(fromString<InstrumentUnit>("thread"), InstrumentUnit::Thread);
}

} // namespace

//===- InstrumentationTest.cpp - Unit tests for instrumentation utilities -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/Instrumentation.h"
#include "llvm/ADT/StringExtras.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

template <typename T> SmallVector<T, 1> vec(std::initializer_list<T> list) {
  return SmallVector<T, 1>(list);
}

TEST(KitInstrumentation, fromStringInstrumentKind) {
  EXPECT_EQ(fromString<InstrumentKind>(""), std::nullopt);
  EXPECT_EQ(fromString<InstrumentKind>("PAPI"), std::nullopt);
  EXPECT_EQ(fromString<InstrumentKind>("counter"), std::nullopt);

  EXPECT_EQ(fromString<InstrumentKind>("generic"), InstrumentKind::Generic);
  EXPECT_EQ(fromString<InstrumentKind>("papi"), InstrumentKind::PAPI);
  EXPECT_EQ(fromString<InstrumentKind>("timer"), InstrumentKind::Timer);
}

TEST(KitInstrumentation, fromStringInstrumentUnit) {
  EXPECT_EQ(fromString<InstrumentUnit>(""), std::nullopt);
  EXPECT_EQ(fromString<InstrumentUnit>("Loop"), std::nullopt);
  EXPECT_EQ(fromString<InstrumentUnit>("function"), std::nullopt);

  EXPECT_EQ(fromString<InstrumentUnit>("loop"), InstrumentUnit::Loop);
  EXPECT_EQ(fromString<InstrumentUnit>("thread"), InstrumentUnit::Thread);
}

TEST(KitInstrumentation, enabled) {
  auto opts = [](InstrumentKind kind) -> KitInstrOptions {
    KitInstrOptions opts;
    opts.addKind(kind);
    return opts;
  };

  EXPECT_FALSE(KitInstrOptions().enabled());
  EXPECT_TRUE(opts(InstrumentKind::Generic).enabled());
  EXPECT_TRUE(opts(InstrumentKind::PAPI).enabled());
  EXPECT_TRUE(opts(InstrumentKind::Timer).enabled());
}

TEST(KitInstrumentation, enabledKind) {
  KitInstrOptions opts;
  EXPECT_FALSE(opts.enabled(InstrumentKind::Generic));
  EXPECT_FALSE(opts.enabled(InstrumentKind::PAPI));
  EXPECT_FALSE(opts.enabled(InstrumentKind::Timer));

  opts.addKind(InstrumentKind::Generic);
  opts.addKind(InstrumentKind::Timer);

  EXPECT_TRUE(opts.enabled(InstrumentKind::Generic));
  EXPECT_FALSE(opts.enabled(InstrumentKind::PAPI));
  EXPECT_TRUE(opts.enabled(InstrumentKind::Timer));
}

TEST(KitInstrumentation, shouldInstrument) {
  KitInstrOptions opts;

  // No names have been added. Any name is legal.
  EXPECT_TRUE(opts.shouldInstrument(""));
  EXPECT_TRUE(opts.shouldInstrument("sesotho"));

  // If even one name has been added, only that name should be instrumented.
  opts.addName("ndebele");
  EXPECT_FALSE(opts.shouldInstrument("sesotho"));
  EXPECT_TRUE(opts.shouldInstrument("ndebele"));

  // Names are case-sensitive.
  EXPECT_FALSE(opts.shouldInstrument("Ndebele"));
}

TEST(KitInstrumentation, setUnitsAll) {
  KitInstrOptions opts;
  opts.setUnitsAll();

  EXPECT_TRUE(opts.enabled(InstrumentUnit::Thread));
  EXPECT_TRUE(opts.enabled(InstrumentUnit::Loop));
}

TEST(KitInstrumentation, setUnitsDefault) {
  KitInstrOptions opts;
  opts.setUnitsDefault();

  EXPECT_FALSE(opts.enabled(InstrumentUnit::Thread));
  EXPECT_TRUE(opts.enabled(InstrumentUnit::Loop));
}

TEST(KitInstrumentation, getKinds) {
  KitInstrOptions opts;

  EXPECT_EQ(opts.getKinds(), vec<InstrumentKind>({}));

  opts.addKind(InstrumentKind::PAPI);
  EXPECT_EQ(opts.getKinds(), vec({InstrumentKind::PAPI}));

  opts.addKind(InstrumentKind::Generic);
  EXPECT_EQ(opts.getKinds(),
            vec({InstrumentKind::Generic, InstrumentKind::PAPI}));

  // Adding it again should have no effect.
  opts.addKind(InstrumentKind::Generic);
  EXPECT_EQ(opts.getKinds(),
            vec({InstrumentKind::Generic, InstrumentKind::PAPI}));

  opts.addKind(InstrumentKind::Timer);
  EXPECT_EQ(opts.getKinds(), vec({InstrumentKind::Generic, InstrumentKind::PAPI,
                                  InstrumentKind::Timer}));
}

TEST(KitInstrumentation, getUnits) {
  KitInstrOptions opts;

  // If nothing has been set explicitly, the default set of units will be
  // returned.
  EXPECT_EQ(opts.getUnits(), vec({InstrumentUnit::Loop}));

  // If a unit is added, it will override the defaults.
  opts.addUnit(InstrumentUnit::Thread);
  EXPECT_EQ(opts.getUnits(), vec({InstrumentUnit::Thread}));

  // Adding it again should have no effect.
  opts.addUnit(InstrumentUnit::Thread);
  EXPECT_EQ(opts.getUnits(), vec({InstrumentUnit::Thread}));

  opts.addUnit(InstrumentUnit::Loop);
  EXPECT_EQ(opts.getUnits(),
            vec({InstrumentUnit::Thread, InstrumentUnit::Loop}));
}

} // namespace

//===- MaybeBoolTest.cpp - Tests for the core Tapir types and enums -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Support/MaybeBool.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitMaybeBool, toInt) {
  // The values of the MaybeBool enum must be the following values.
  EXPECT_EQ(int(MaybeBool::Off), 0);
  EXPECT_EQ(int(MaybeBool::On), 1);
  EXPECT_EQ(int(MaybeBool::Any), 3);
}

TEST(KitMaybeBool, fromString) {
  EXPECT_EQ(fromString<MaybeBool>(""), std::nullopt);
  EXPECT_EQ(fromString<MaybeBool>("ON"), std::nullopt);
  EXPECT_EQ(fromString<MaybeBool>("OFF"), std::nullopt);

  EXPECT_EQ(fromString<MaybeBool>("off"), MaybeBool::Off);
  EXPECT_EQ(fromString<MaybeBool>("on"), MaybeBool::On);
  EXPECT_EQ(fromString<MaybeBool>("any"), MaybeBool::Any);
}

} // namespace

//===- ToStringTest.cpp - Unit tests for toString utilities ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Support/ToString.h"
#include "llvm/ADT/SmallVector.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitToString, toStringVec) {
  SmallVector<TTID, 3> vec({TTID::Nolo, TTID::OpenMP, TTID::Serial});

  EXPECT_EQ(toString(SmallVector<TTID, 1>({})), "");
  EXPECT_EQ(toString(vec), "nolo,openmp,serial");
  EXPECT_EQ(toString(vec, ", "), "nolo, openmp, serial");
}

TEST(KitToString, toStringSet) {
  std::set<int> set({1, 2, 3});

  EXPECT_EQ(toString(std::set<int>{}), "");
  EXPECT_EQ(toString(set), "1,2,3");
  EXPECT_EQ(toString(set, "."), "1.2.3");
}

} // namespace

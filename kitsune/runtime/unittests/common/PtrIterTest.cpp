//===- PtrIteratorTest.cpp - Tests for the PtrIterator iterator -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common/ptriter.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <set>
#include <vector>

using namespace kitrt;

namespace {

template <typename C> class Range {
private:
  typename C::const_iterator b, e;

public:
  Range(const C &c) : b(c.begin()), e(c.end()) {}

  PtrIterator<typename C::const_iterator> begin() { return b; }
  PtrIterator<typename C::const_iterator> end() { return e; }
};

TEST(PtrIterator, vecUniq) {
  std::vector<std::unique_ptr<int>> v;
  v.emplace_back(std::make_unique<int>(139));
  v.emplace_back(std::make_unique<int>(167));
  v.emplace_back(std::make_unique<int>(193));

  std::vector<int> actual;
  for (const int &i : Range<decltype(v)>(v))
    actual.push_back(i);

  EXPECT_EQ(actual.size(), 3U);
  EXPECT_EQ(actual[0], 139);
  EXPECT_EQ(actual[1], 167);
  EXPECT_EQ(actual[2], 193);
}

TEST(PtrIterator, vecPtr) {
  int i0 = 239;
  int i1 = 263;
  int i2 = 293;

  std::vector<int *> v = {&i0, &i1, &i2};
  std::vector<int> actual;
  for (const int &i : Range<decltype(v)>(v))
    actual.push_back(i);

  EXPECT_EQ(actual.size(), 3U);
  EXPECT_EQ(actual[0], 239);
  EXPECT_EQ(actual[1], 263);
  EXPECT_EQ(actual[2], 293);
}

TEST(PtrIterator, setPtr) {
  int i0 = 331;
  int i1 = 313;
  int i2 = 367;

  std::set<int *> v = {&i0, &i1, &i2};
  std::vector<int> actual;
  for (const int &i : Range<decltype(v)>(v))
    actual.push_back(i);

  std::sort(actual.begin(), actual.end());

  EXPECT_EQ(actual.size(), 3U);
  EXPECT_EQ(actual[0], 313);
  EXPECT_EQ(actual[1], 331);
  EXPECT_EQ(actual[2], 367);
}

} // namespace

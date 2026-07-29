//===- TypeTraitsTest.cpp - Unit tests for type traits --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Support/TypeTraits.h"

#include "gtest/gtest.h"

namespace {

using Bool = bool;

TEST(KitTypeTraits, isBool) {
  EXPECT_TRUE(std::is_bool_v<bool>);
  EXPECT_TRUE(std::is_bool_v<const bool>);
  EXPECT_TRUE(std::is_bool_v<Bool>);

  EXPECT_FALSE(std::is_bool_v<bool &>);
  EXPECT_FALSE(std::is_bool_v<const bool &>);

  EXPECT_FALSE(std::is_bool_v<int8_t>);
  EXPECT_FALSE(std::is_bool_v<uint8_t>);
  EXPECT_FALSE(std::is_bool_v<unsigned char>);
}

TEST(KitTypeTraits, isInteger) {
  EXPECT_TRUE(std::is_integer_v<int8_t>);
  EXPECT_TRUE(std::is_integer_v<int16_t>);
  EXPECT_TRUE(std::is_integer_v<int32_t>);
  EXPECT_TRUE(std::is_integer_v<int64_t>);
  EXPECT_TRUE(std::is_integer_v<uint8_t>);
  EXPECT_TRUE(std::is_integer_v<uint16_t>);
  EXPECT_TRUE(std::is_integer_v<uint32_t>);
  EXPECT_TRUE(std::is_integer_v<uint64_t>);

  EXPECT_TRUE(std::is_integer_v<char>);
  EXPECT_TRUE(std::is_integer_v<signed char>);
  EXPECT_TRUE(std::is_integer_v<unsigned char>);
  EXPECT_TRUE(std::is_integer_v<short>);
  EXPECT_TRUE(std::is_integer_v<int>);
  EXPECT_TRUE(std::is_integer_v<long>);
  EXPECT_TRUE(std::is_integer_v<long long>);
  EXPECT_TRUE(std::is_integer_v<unsigned>);
  EXPECT_TRUE(std::is_integer_v<unsigned char>);
  EXPECT_TRUE(std::is_integer_v<unsigned short>);
  EXPECT_TRUE(std::is_integer_v<unsigned int>);
  EXPECT_TRUE(std::is_integer_v<unsigned long>);
  EXPECT_TRUE(std::is_integer_v<unsigned long long>);

  EXPECT_TRUE(std::is_integer_v<const int8_t>);
  EXPECT_TRUE(std::is_integer_v<const int16_t>);
  EXPECT_TRUE(std::is_integer_v<const int32_t>);
  EXPECT_TRUE(std::is_integer_v<const int64_t>);
  EXPECT_TRUE(std::is_integer_v<const uint8_t>);
  EXPECT_TRUE(std::is_integer_v<const uint16_t>);
  EXPECT_TRUE(std::is_integer_v<const uint32_t>);
  EXPECT_TRUE(std::is_integer_v<const uint64_t>);

  EXPECT_TRUE(std::is_integer_v<const char>);
  EXPECT_TRUE(std::is_integer_v<const signed char>);
  EXPECT_TRUE(std::is_integer_v<const unsigned char>);
  EXPECT_TRUE(std::is_integer_v<const short>);
  EXPECT_TRUE(std::is_integer_v<const int>);
  EXPECT_TRUE(std::is_integer_v<const long>);
  EXPECT_TRUE(std::is_integer_v<const long long>);
  EXPECT_TRUE(std::is_integer_v<const unsigned>);
  EXPECT_TRUE(std::is_integer_v<const unsigned char>);
  EXPECT_TRUE(std::is_integer_v<const unsigned short>);
  EXPECT_TRUE(std::is_integer_v<const unsigned int>);
  EXPECT_TRUE(std::is_integer_v<const unsigned long>);
  EXPECT_TRUE(std::is_integer_v<const unsigned long long>);

  EXPECT_FALSE(std::is_integer_v<int8_t &>);
  EXPECT_FALSE(std::is_integer_v<int16_t &>);
  EXPECT_FALSE(std::is_integer_v<int32_t &>);
  EXPECT_FALSE(std::is_integer_v<int64_t &>);
  EXPECT_FALSE(std::is_integer_v<uint8_t &>);
  EXPECT_FALSE(std::is_integer_v<uint16_t &>);
  EXPECT_FALSE(std::is_integer_v<uint32_t &>);
  EXPECT_FALSE(std::is_integer_v<uint64_t &>);

  EXPECT_FALSE(std::is_integer_v<bool>);
  EXPECT_FALSE(std::is_integer_v<const bool>);
  EXPECT_FALSE(std::is_integer_v<Bool>);
}

struct Iterable {
  int begin();
  int end();
};

struct NotIterableB {
  int begin();
};

struct NotIterableE {
  int end();
};

struct NotIterable2 {};

TEST(KitTypeTraits, isIterable) {
  EXPECT_FALSE(std::is_iterable_v<NotIterable2>);
  EXPECT_FALSE(std::is_iterable_v<NotIterableB>);
  EXPECT_FALSE(std::is_iterable_v<NotIterableE>);
  EXPECT_TRUE(std::is_iterable_v<Iterable>);
}

} // namespace

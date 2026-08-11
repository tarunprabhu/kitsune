//===- TraitsTest.cpp - Tests for type traits -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common/traits.h"

#include "gtest/gtest.h"

namespace {

class StructEmpty {};
class StructForward;

TEST(KitrtTraits, isComplete) {
  // Builtin types are always complete ...
  EXPECT_TRUE(std::is_complete_v<std::nullptr_t>);
  EXPECT_TRUE(std::is_complete_v<int>);
  EXPECT_TRUE(std::is_complete_v<void *>);

  // ... except void because we cannot do sizeof(void).
  EXPECT_FALSE(std::is_complete_v<void>);

  // Empty structs are complete too.
  EXPECT_TRUE(std::is_complete_v<StructEmpty>);

  // Arrays that do not specify the number of elements are incomplete, by
  // definition. But those with a number of elements, are complete.
  EXPECT_FALSE(std::is_complete_v<int[]>);
  EXPECT_TRUE(std::is_complete_v<int[1]>);

  // This is the case that we actually care about.
  EXPECT_FALSE(std::is_complete_v<StructForward>);
}

} // namespace

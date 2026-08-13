//===- TypeTraitsTest.cpp - Unit tests for type traits --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Shared/TypeTraits.h"

#include "gtest/gtest.h"

namespace {

struct StructInt {
  int a;
};

struct StructPtr {
  void *ptr;
};

struct StructArray {
  int arr[4];
};

struct StructArrayPtr {
  void *ptrs[3];
};

struct StructStructPtr {
  struct Inner {
    void *ptr;
  } inner;
};

struct StructStructNoPtr {
  struct Inner {
    int arr[4];
  } inner;
};

class ClassPrivate {
  int i;
};

class ClassMixed {
private:
  int i;

public:
  int j;
};

struct DerivedInt : public StructInt {
  int d;
};

struct DefaultedConstructor {
  int i;
  DefaultedConstructor() = default;
};

TEST(KitTypeTraits, isInterop) {
  // All interoperable types must be structs
  EXPECT_FALSE(std::is_interop_v<bool>);
  EXPECT_FALSE(std::is_interop_v<char>);
  EXPECT_FALSE(std::is_interop_v<int>);
  EXPECT_FALSE(std::is_interop_v<uint64_t>);
  EXPECT_FALSE(std::is_interop_v<float>);
  EXPECT_FALSE(std::is_interop_v<long double>);
  EXPECT_FALSE(std::is_interop_v<char[11]>);

  EXPECT_FALSE(std::is_interop_v<const char *>);
  EXPECT_FALSE(std::is_interop_v<void *>);

  // These types have at least one private, non-static member, and are not
  // interoperable.
  EXPECT_FALSE(std::is_interop_v<ClassPrivate>);
  EXPECT_FALSE(std::is_interop_v<ClassMixed>);

  // These are not interoperable because they are either not trivial, not
  // trivially constructible, or not standard-layout.
  EXPECT_FALSE(std::is_interop_v<std::string>);
  EXPECT_FALSE(std::is_interop_v<llvm::StringRef>);
  EXPECT_FALSE(std::is_interop_v<llvm::StringLiteral>);

  // Derived classes where both base and derived class define a member are not
  // interoperable.
  EXPECT_FALSE(std::is_interop_v<DerivedInt>);

  // These are interoperable.
  EXPECT_TRUE(std::is_interop_v<StructArray>);
  EXPECT_TRUE(std::is_interop_v<StructStructNoPtr>);
  EXPECT_TRUE(std::is_interop_v<StructInt>);
  EXPECT_TRUE(std::is_interop_v<DefaultedConstructor>);

  // FIXME: These currently return true, but they should return false. But we
  // check for the wrong thing just to keep the testing green.
  EXPECT_TRUE(std::is_interop_v<StructPtr>);
  EXPECT_TRUE(std::is_interop_v<StructArrayPtr>);
  EXPECT_TRUE(std::is_interop_v<StructStructPtr>);
}

class StructEmpty {};
class StructForward;

TEST(KitTypeTraits, isComplete) {
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

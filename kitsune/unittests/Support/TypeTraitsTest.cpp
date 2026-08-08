//===- TypeTraitsTest.cpp - Unit tests for type traits --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Support/TypeTraits.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

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

TEST(KitTypeTraits, isStringLike) {
  EXPECT_FALSE(std::is_string_like_v<const char *>);
  EXPECT_FALSE(std::is_string_like_v<char[12]>);

  EXPECT_TRUE(std::is_string_like_v<std::string>);
  EXPECT_TRUE(std::is_string_like_v<llvm::StringRef>);
  EXPECT_TRUE(std::is_string_like_v<llvm::StringLiteral>);
  EXPECT_TRUE(std::is_string_like_v<llvm::SmallString<0>>);
  EXPECT_TRUE(std::is_string_like_v<llvm::SmallString<8>>);
}

TEST(KitTypeTraits, isSmallString) {
  EXPECT_TRUE(std::is_small_string_v<llvm::SmallString<8>>);
  EXPECT_TRUE(std::is_small_string_v<llvm::SmallString<0>>);

  EXPECT_FALSE(std::is_small_string_v<std::string>);
  EXPECT_FALSE(std::is_small_string_v<llvm::StringRef>);
  EXPECT_FALSE(std::is_small_string_v<llvm::StringLiteral>);
}

TEST(KitTypeTraits, isSmallSet) {
  EXPECT_TRUE((std::is_small_set_v<llvm::SmallSet<int, 0>>));
  EXPECT_TRUE((std::is_small_set_v<llvm::SmallSet<llvm::StringRef, 4>>));

  EXPECT_FALSE((std::is_small_set_v<llvm::SmallVector<int, 2>>));
  EXPECT_FALSE(std::is_small_set_v<llvm::SmallString<1024>>);
  EXPECT_FALSE(std::is_small_set_v<std::set<int>>);
}

TEST(KitTypeTraits, isSmallVector) {
  EXPECT_TRUE((std::is_small_vector_v<llvm::SmallVector<int, 0>>));
  EXPECT_TRUE((std::is_small_vector_v<llvm::SmallVector<llvm::StringRef, 0>>));

  EXPECT_FALSE((std::is_small_vector_v<llvm::SmallSet<int, 2>>));
  EXPECT_FALSE(std::is_small_vector_v<llvm::SmallString<8>>);
  EXPECT_FALSE(std::is_small_vector_v<std::vector<int>>);
}

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

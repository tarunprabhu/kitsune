//==- TestValues.h - Utilities to get "random" values for tests --*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_UNITTEST_CORE_TEST_VALUES_H
#define KITSUNE_UNITTEST_CORE_TEST_VALUES_H

#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"

// Nothing here generates truly random values, though we could. It's almost
// certainly not worth the trouble. Instead, we use the index to lookup from a
// fixed set of "random" values. These are only really useful for scalar values
// and strings. We explicitly instantiate all enums because there is no
// guarantee what values will be valid for the enums.

template <typename T, std::enable_if_t<
                          std::is_same_v<T, llvm::TapirSpawnStrategy>, int> = 0>
T get(unsigned idx) {
  static constexpr T pool[] = {T::Sequential, T::Basic};
  return pool[idx % 2];
}

template <typename T, std::enable_if_t<std::is_same_v<T, llvm::TTID>, int> = 0>
T get(unsigned idx) {
  static constexpr T pool[] = {T::Cuda, T::Hip};
  return pool[idx % 2];
}

template <typename T,
          std::enable_if_t<
              std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>, int> = 0>
T get(unsigned idx) {
  static constexpr T pool[] = {
      2, 13, 73, 167, 179, 181, 199, 211,
  };
  return pool[idx % (sizeof(pool) / sizeof(T))];
}

template <typename T, std::enable_if_t<std::is_same_v<T, int16_t> ||
                                           std::is_same_v<T, uint16_t>,
                                       int> = 0>
T get(unsigned idx) {
  static constexpr T pool[] = {
      373, 1103, 4409, 7793, 11939, 18253, 21169, 31219,
  };
  return pool[idx % (sizeof(pool) / sizeof(T))];
}

template <typename T, std::enable_if_t<std::is_same_v<T, int32_t> ||
                                           std::is_same_v<T, uint32_t>,
                                       int> = 0>
T get(unsigned idx) {
  static constexpr T pool[] = {
      63949, 65713, 69313, 73009, 76801, 84673, 106033, 108301,
  };
  return pool[idx % (sizeof(pool) / sizeof(T))];
}

template <typename T, std::enable_if_t<std::is_same_v<T, int64_t> ||
                                           std::is_same_v<T, uint64_t>,
                                       int> = 0>
T get(unsigned idx) {
  static constexpr T pool[] = {
      17179869143L,  34359738319L,  68719476713L,   137438953403L,
      274877906791L, 549755813669L, 1099511627477L, 2199023255413L,
  };
  return pool[idx % (sizeof(pool) / sizeof(T))];
}

template <typename T, std::enable_if_t<std::is_same_v<T, float>, int> = 0>
T get(unsigned idx) {
  static constexpr T pool[] = {
      3.14159F,  2.71828F, 1.61803F, 0.57721F,
      0.207879F, 0.01101F, 4.66920F, 0.91596F,
  };
  return pool[idx % (sizeof(pool) / sizeof(T))];
}

template <typename T, std::enable_if_t<std::is_same_v<T, double>, int> = 0>
T get(unsigned idx) {
  static constexpr T pool[] = {
      3.14159265358979323, 2.71828182845904523, 1.61803398874989484,
      0.57721566490153286, 0.12345678910111213, 0.01101001100101101,
      4.69920160910299067, 0.91596559417721901,
  };
  return pool[idx % (sizeof(pool) / sizeof(T))];
}

template <typename T,
          std::enable_if_t<std::is_same_v<T, llvm::StringRef>, int> = 0>
T get(unsigned idx) {
  static constexpr T pool[] = {
      "heffalump", "woozle", "jagular", "backson",
      "tigger",    "eeyore", "kanga",   "haycorn",
  };
  return pool[idx % (sizeof(pool) / sizeof(T))];
}

#endif // KITSUNE_UNITTEST_CORE_TEST_VALUES_H

//==- CheckUtils.h - Utilities for gtest checks in the presence of errors --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// For the object-related unit tests, we compress a binary, then encode it in
// base64 so the raw object data can be included in the source files. This
// provides utilities to decompress such objects.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_UNITTESTS_OBJECT_CHECK_UTILS_H
#define KITSUNE_UNITTESTS_OBJECT_CHECK_UTILS_H

#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

namespace llvm {

namespace detail {

static void check_true(Expected<bool> val) {
  EXPECT_TRUE((bool)val);
  EXPECT_TRUE(*val);
}

static void check_false(Expected<bool> val) {
  EXPECT_TRUE((bool)val);
  EXPECT_FALSE(*val);
}

template <typename T, typename U,
          std::enable_if_t<std::is_convertible_v<T, U> ||
                               (std::is_integral_v<T> && std::is_integral_v<U>),
                           int> = 0>
void check_eq(Expected<T> val, const U &expected) {
  EXPECT_TRUE((bool)val);
  EXPECT_EQ(*val, static_cast<T>(expected));
}

} // namespace detail

} // namespace llvm

#endif // KITSUNE_UNITTESTS_OBJECT_CHECK_UTILS_H

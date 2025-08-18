//=- CheckUtils.cpp - Utilities for gtest checks in the presence of errors --=//
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

#include "CheckUtils.h"

#include "gtest/gtest.h"

using namespace llvm;

void llvm::detail::check_true(Expected<bool> val) {
  EXPECT_TRUE((bool)val);
  EXPECT_TRUE(*val);
}

void llvm::detail::check_false(Expected<bool> val) {
  EXPECT_TRUE((bool)val);
  EXPECT_FALSE(*val);
}

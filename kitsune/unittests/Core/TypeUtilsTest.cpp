//===- TypeUtilsTest.cpp - Tests for Kitsune's type utilities -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/TypeUtils.h"

#include "llvm/IR/LLVMContext.h"

#include "gtest/gtest.h"

using namespace llvm;

TEST(TypeUtils, getLLVMTypeFor) {
  LLVMContext ctx;

  EXPECT_TRUE(getLLVMTypeFor<int8_t>(ctx)->isIntegerTy(8));
  EXPECT_TRUE(getLLVMTypeFor<uint8_t>(ctx)->isIntegerTy(8));

  EXPECT_TRUE(getLLVMTypeFor<int16_t>(ctx)->isIntegerTy(16));
  EXPECT_TRUE(getLLVMTypeFor<uint16_t>(ctx)->isIntegerTy(16));
  EXPECT_TRUE(getLLVMTypeFor<short>(ctx)->isIntegerTy(16));
  EXPECT_TRUE(getLLVMTypeFor<unsigned short>(ctx)->isIntegerTy(16));

  EXPECT_TRUE(getLLVMTypeFor<int32_t>(ctx)->isIntegerTy(32));
  EXPECT_TRUE(getLLVMTypeFor<uint32_t>(ctx)->isIntegerTy(32));
  EXPECT_TRUE(getLLVMTypeFor<int>(ctx)->isIntegerTy(32));
  EXPECT_TRUE(getLLVMTypeFor<unsigned int>(ctx)->isIntegerTy(32));

  EXPECT_TRUE(getLLVMTypeFor<int64_t>(ctx)->isIntegerTy(64));
  EXPECT_TRUE(getLLVMTypeFor<uint64_t>(ctx)->isIntegerTy(64));

  EXPECT_TRUE(getLLVMTypeFor<float>(ctx)->isFloatTy());
  EXPECT_TRUE(getLLVMTypeFor<double>(ctx)->isDoubleTy());

  EXPECT_TRUE(getLLVMTypeFor<void *>(ctx)->isPointerTy());
  EXPECT_TRUE(getLLVMTypeFor<int *>(ctx)->isPointerTy());
  EXPECT_TRUE(getLLVMTypeFor<const char *>(ctx)->isPointerTy());
  EXPECT_TRUE(getLLVMTypeFor<double *>(ctx)->isPointerTy());

  EXPECT_TRUE(getLLVMTypeFor<const int32_t>(ctx)->isIntegerTy(32));
  EXPECT_TRUE(getLLVMTypeFor<const uint32_t>(ctx)->isIntegerTy(32));

#if !defined(__sun)
  EXPECT_TRUE(getLLVMTypeFor<char>(ctx)->isIntegerTy(8));
  EXPECT_TRUE(getLLVMTypeFor<signed char>(ctx)->isIntegerTy(8));
  EXPECT_TRUE(getLLVMTypeFor<unsigned char>(ctx)->isIntegerTy(8));
#endif // __sun

  // On some systems, long long and int64_t are the same type. On those systems,
  // we do not instantiate long and long long. __unix__ below implies __linux__
  // and *BSD, but not MacOSX.
#if __unix__
  EXPECT_TRUE(getLLVMTypeFor<long>(ctx)->isIntegerTy(64));
  EXPECT_TRUE(getLLVMTypeFor<unsigned long>(ctx)->isIntegerTy(64));
  EXPECT_TRUE(getLLVMTypeFor<long long>(ctx)->isIntegerTy(64));
  EXPECT_TRUE(getLLVMTypeFor<unsigned long long>(ctx)->isIntegerTy(64));
#endif // __unix__
}

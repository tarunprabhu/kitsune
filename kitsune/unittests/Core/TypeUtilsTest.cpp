//===- TypeUtilsTest.cpp - Tests for Kitsune's type utilities -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/TypeUtils.h"
#include "kitsune/Core/AddrSpace.h"

#include "llvm/IR/LLVMContext.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitTypeUtils, isMobilePointerTy) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  Type *i32 = Type::getInt32Ty(ctx);
  Type *ptr = PointerType::getUnqual(ctx);
  Type *mobile = PointerType::get(ctx, KitAS::Mobile);
  Type *ptrAS = PointerType::get(ctx, 1);

  EXPECT_FALSE(isMobilePointerTy(voidTy));
  EXPECT_FALSE(isMobilePointerTy(i32));
  EXPECT_FALSE(isMobilePointerTy(ptr));
  EXPECT_FALSE(isMobilePointerTy(ptrAS));

  EXPECT_TRUE(isMobilePointerTy(mobile));
}

TEST(KitTypeUtils, getLLVMTypeFor) {
  LLVMContext ctx;

  EXPECT_TRUE(getLLVMTypeFor<bool>(ctx)->isIntegerTy(1));
  EXPECT_TRUE(getLLVMTypeFor<const bool>(ctx)->isIntegerTy(1));

  EXPECT_TRUE(getLLVMTypeFor<int8_t>(ctx)->isIntegerTy(8));
  EXPECT_TRUE(getLLVMTypeFor<uint8_t>(ctx)->isIntegerTy(8));
  EXPECT_TRUE(getLLVMTypeFor<char>(ctx)->isIntegerTy(8));
  EXPECT_TRUE(getLLVMTypeFor<signed char>(ctx)->isIntegerTy(8));
  EXPECT_TRUE(getLLVMTypeFor<unsigned char>(ctx)->isIntegerTy(8));

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
  EXPECT_TRUE(getLLVMTypeFor<const float>(ctx)->isFloatTy());
  EXPECT_TRUE(getLLVMTypeFor<double>(ctx)->isDoubleTy());
  EXPECT_TRUE(getLLVMTypeFor<const double>(ctx)->isDoubleTy());

  EXPECT_TRUE(getLLVMTypeFor<void *>(ctx)->isPointerTy());
  EXPECT_TRUE(getLLVMTypeFor<int *>(ctx)->isPointerTy());
  EXPECT_TRUE(getLLVMTypeFor<const char *>(ctx)->isPointerTy());
  EXPECT_TRUE(getLLVMTypeFor<double *>(ctx)->isPointerTy());

  EXPECT_TRUE(getLLVMTypeFor<const int32_t>(ctx)->isIntegerTy(32));
  EXPECT_TRUE(getLLVMTypeFor<const uint32_t>(ctx)->isIntegerTy(32));

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

struct S0 {};

struct S3 {
  char s[3];
};

struct S16 {
  int64_t m0;
  int32_t m1;
};

struct S24 {
  char m0;
  double m1;
  uint16_t m2;
  float m3;
};

TEST(KitTypeUtils, getLLVMByteArrayTypeFor) {
  LLVMContext ctx;
  Type *i8 = Type::getInt8Ty(ctx);

  // Empty struct still return a sizeof(1).
  ArrayType *a0 = getLLVMByteArrayTypeFor<S0>(ctx);
  EXPECT_EQ(a0->getElementType(), i8);
  EXPECT_EQ(a0->getNumElements(), 1U);

  // sizeof does not necessarily "align up".
  ArrayType *a3 = getLLVMByteArrayTypeFor<S3>(ctx);
  EXPECT_EQ(a3->getElementType(), i8);
  EXPECT_EQ(a3->getNumElements(), 3U);

  // Padding is accounted for when computing sizeof.
  ArrayType *a16 = getLLVMByteArrayTypeFor<S16>(ctx);
  EXPECT_EQ(a16->getElementType(), i8);
  EXPECT_EQ(a16->getNumElements(), 16U);

  ArrayType *a24 = getLLVMByteArrayTypeFor<S24>(ctx);
  EXPECT_EQ(a24->getElementType(), i8);
  EXPECT_EQ(a24->getNumElements(), 24U);
}

} // namespace

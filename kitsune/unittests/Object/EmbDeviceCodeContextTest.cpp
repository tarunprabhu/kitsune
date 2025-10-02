//=- EmbDeviceCodeContextTest.cpp - Tests for embedded device code context --=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Object/EmbDeviceCodeContext.h"
#include "CheckUtils.h"
#include "CompressedBinary.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

static std::unique_ptr<ObjectFile> o(const CompressedBinary &c) {
  Expected<std::unique_ptr<ObjectFile>> obj =
      ObjectFile::createObjectFile(c.memBuf);
  if (not obj)
    ADD_FAILURE();
  return std::move(*obj);
}

// A relocatable ELF object containing device for the sm_72 cuda architecture.
static const CompressedBinary sm72(
    "file1",
    "eNqrd/VxY2JkZIABRgY7BgQPATSYEWwHKMkMVMnDwATm1UPMMWaHm7OPoRGLOR5Qwz1YPYAm"
    "WIDNYQPZqFecUVxSVJKYBGTB6MpcJDq+OCMvpYJBL69MLzMvLZ9BLyU1qTQ9Pq0oMTcVLJqc"
    "mJOTXpRYkAHmFRTll+SXVBak0tRonMAeiJkZWLHK/QcCEP0PSv+F0n+gNCWAEWwvJnCA0tFY"
    "1CMDbhz6ZxOpXxiImbDo/wGlDaA0ExRzQPkSUNoIi5ng9MeI27/o4Q4UK8ClXwHKh/mRBUrD"
    "3IFIItmZJXrJ+SmpesmlKYkMtIoXAWZU/+PyFydUjAlHvF5Ak+BAUwcAEcVkjw==",
    1000);

// A relocatable ELF object containing device for the sm_72 cuda architecture.
// This is a duplicate of sm72 above, but with a different file name.
static const CompressedBinary sm72_2(
    "file3",
    "eNqrd/VxY2JkZIABRgY7BgQPATSYEWwHKMkMVMnDwATm1UPMMWaHm7OPoRGLOR5Qwz1YPYAm"
    "WIDNYQPZqFecUVxSVJKYBGTB6MpcJDq+OCMvpYJBL69MLzMvLZ9BLyU1qTQ9Pq0oMTcVLJqc"
    "mJOTXpRYkAHmFRTll+SXVBak0tRonMAeiJkZWLHK/QcCEP0PSv+F0n+gNCWAEWwvJnCA0tFY"
    "1CMDbhz6ZxOpXxiImbDo/wGlDaA0ExRzQPkSUNoIi5ng9MeI27/o4Q4UK8ClXwHKh/mRBUrD"
    "3IFIItmZJXrJ+SmpesmlKYkMtIoXAWZU/+PyFydUjAlHvF5Ak+BAUwcAEcVkjw==",
    1000);

// A relocatable ELF object containing device for the sm_80 cuda architecture.
static const CompressedBinary sm80(
    "file2",
    "eNqrd/VxY2JkZIABRgY7BgQPATSYEWwHKMkMVMnHwATm1UPMMWaHm7OPoRGLOR5QwwNYA4Am"
    "WIDNYQPZqFecUVxSVJKYBGTB6MpcJDq+OCMvpYJBL69MLzMvLZ9BLyU1qTQ9Pq0oMTcVLJqc"
    "mJOTXpRYkAHmFRTll+SXVBak0tRonMAeiJkZWLHK/QcCEP0PSv+F0n+gNCWAEWwvJnCA0tFY"
    "1CMDbhz6ZxOpXxiImbDo/wGlDaA0ExRzQPkSUNoIi5ng9MeI27/o4Q4UK8ClXwHKh/mRBUrD"
    "3IFIItmZJXrJ+SmpesmlKYkMtIoXAWZU/+PyFydUjAlHvF5Ak+BAUwcAUAVkoQ==",
    1000);

TEST(EmbDeviceCodeContextTest, init) {
  EmbDeviceCodeContext ctx;

  EXPECT_TRUE(ctx.empty());
  EXPECT_TRUE(ctx.getTTIDs().empty());
  EXPECT_TRUE(ctx.get(TTID::Cuda).empty());
  EXPECT_TRUE(ctx.get(TTID::Hip).empty());
  EXPECT_TRUE(ctx.get(TTID::Serial).empty());

  detail::check_err(ctx.getEmbDeviceCodeId(TTID::Cuda));
  detail::check_err(ctx.getEmbDeviceCodeId(TTID::Hip));
  detail::check_false(ctx.contains(*elfEmpty));
}

TEST(EmbDeviceCodeContextTest, empty) {
  EmbDeviceCodeContext ctx;
  const std::unique_ptr<ObjectFile> bin72 = o(sm72);

  Expected<unsigned> res = ctx.add(cast<Binary>(*bin72));
  detail::check_eq(std::move(res), 1);
  EXPECT_FALSE(ctx.empty());
}

TEST(EmbDeviceCodeContextTest, getEmbDeviceCodeIdFail) {
  EmbDeviceCodeContext ctx;
  const std::unique_ptr<ObjectFile> bin72 = o(sm72);
  const std::unique_ptr<ObjectFile> bin80 = o(sm80);

  Expected<unsigned> res1 = ctx.add(cast<Binary>(*bin72));
  detail::check_eq(std::move(res1), 1);

  Expected<unsigned> res2 = ctx.add(cast<Binary>(*bin80));
  detail::check_eq(std::move(res2), 1);

  detail::check_err(ctx.getEmbDeviceCodeId(TTID::Cuda));
}

TEST(EmbdeviceCodeContextTest, getEmbDeviceCodeIdPass) {
  EmbDeviceCodeContext ctx;
  const std::unique_ptr<ObjectFile> bin1 = o(sm72);
  const std::unique_ptr<ObjectFile> bin2 = o(sm72_2);

  Expected<unsigned> res1 = ctx.add(cast<Binary>(*bin1));
  EXPECT_TRUE(bool(res1));
  EXPECT_EQ(*res1, 1U);

  Expected<unsigned> res2 = ctx.add(cast<Binary>(*bin2));
  EXPECT_TRUE(bool(res2));
  EXPECT_EQ(*res2, 1U);

  Expected<EmbDeviceCode::Id> tt = ctx.getEmbDeviceCodeId(TTID::Cuda);
  EXPECT_TRUE(bool(tt));
  EXPECT_EQ(*tt, EmbDeviceCode::SM_72);
}

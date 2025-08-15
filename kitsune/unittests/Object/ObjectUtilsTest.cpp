//===- ObjectUtilsTest.cpp - Tests for Kitsune's object file utilities ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Object/ObjectUtils.h"
#include "CheckUtils.h"
#include "CompressedBinary.h"
#include "kitsune/Object/EmbDeviceCodeContext.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

TEST(ObjectUtilsTest, hasEmbDeviceCode) {
  detail::check_false(hasEmbDeviceCode(*elfEmpty));
  detail::check_true(hasEmbDeviceCode(*elfCuda));
  detail::check_true(hasEmbDeviceCode(*elfHip));
  detail::check_true(hasEmbDeviceCode(*elfMulti));
}

TEST(ObjectUtilsTest, getEmbDeviceCodeTTIDs) {
  using Vec = SmallVector<TTID, 0>;

  detail::check_eq(getEmbDeviceCodeTTIDs(*elfEmpty), Vec({}));
  detail::check_eq(getEmbDeviceCodeTTIDs(*elfCuda), Vec({TTID::Cuda}));
  detail::check_eq(getEmbDeviceCodeTTIDs(*elfHip), Vec({TTID::Hip}));

  Expected<SmallVector<TTID, 0>> tts = getEmbDeviceCodeTTIDs(*elfMulti);
  EXPECT_TRUE(bool(tts));

  std::sort(tts->begin(), tts->end());
  EXPECT_EQ(*tts, Vec({TTID::Cuda, TTID::Hip}));
}

// The add* tests actually test EmbDeviceCodeContext::add(ObjectFile). But these
// are tested here since the are defined in kitsune/lib/Object/ObjectUtils.cpp.
TEST(ObjectUtilsTest, addEmpty) {
  EmbDeviceCodeContext ctx;
  Expected<unsigned> res = ctx.add(cast<Binary>(*elfEmpty));

  EXPECT_TRUE(bool(res));
  EXPECT_TRUE(*res == 0);
  EXPECT_TRUE(ctx.getTTIDs().empty());
  EXPECT_FALSE(ctx.contains(*elfEmpty));
}

TEST(ObjectUtilsTest, addCuda) {
  EmbDeviceCodeContext ctx;
  SmallVector<TTID, 2> tts = {TTID::Cuda};

  Expected<unsigned> res1 = ctx.add(cast<Binary>(*elfCuda));
  EXPECT_TRUE(bool(res1));
  EXPECT_EQ(*res1, 1U);
  EXPECT_EQ(ctx.getTTIDs(), tts);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 1UL);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 0UL);
  EXPECT_TRUE(ctx.contains(*elfCuda));

  Expected<unsigned> res2 = ctx.add(cast<Binary>(*elfCuda));
  EXPECT_TRUE(bool(res2));
  EXPECT_EQ(*res2, 0U);
  EXPECT_EQ(ctx.getTTIDs(), tts);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 1UL);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 0UL);
}

TEST(ObjectUtilsTest, addHip) {
  EmbDeviceCodeContext ctx;
  SmallVector<TTID, 2> tts = {TTID::Hip};

  Expected<unsigned> res1 = ctx.add(cast<Binary>(*elfHip));
  EXPECT_TRUE(bool(res1));
  EXPECT_EQ(*res1, 1U);
  EXPECT_EQ(ctx.getTTIDs(), tts);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 0UL);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 1UL);
  EXPECT_TRUE(ctx.contains(*elfHip));

  Expected<unsigned> res2 = ctx.add(cast<Binary>(*elfHip));
  EXPECT_TRUE(bool(res2));
  EXPECT_EQ(*res2, 0U);
  EXPECT_EQ(ctx.getTTIDs(), tts);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 0UL);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 1UL);
}

TEST(ObjectUtilsTest, addMulti) {
  EmbDeviceCodeContext ctx;
  SmallVector<TTID, 2> tts = {TTID::Cuda, TTID::Hip};

  Expected<unsigned> res1 = ctx.add(cast<Binary>(*elfMulti));
  EXPECT_TRUE(bool(res1));
  EXPECT_EQ(*res1, 2U);
  EXPECT_EQ(ctx.getTTIDs(), tts);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 1U);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 1U);
  EXPECT_TRUE(ctx.contains(*elfMulti));

  Expected<unsigned> res2 = ctx.add(cast<Binary>(*elfCuda));
  EXPECT_TRUE(bool(res2));
  EXPECT_EQ(*res2, 1U);
  EXPECT_EQ(ctx.getTTIDs(), tts);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 2U);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 1U);
  EXPECT_TRUE(ctx.contains(*elfCuda));

  Expected<unsigned> res3 = ctx.add(cast<Binary>(*elfHip));
  EXPECT_TRUE(bool(res3));
  EXPECT_EQ(*res3, 1U);
  EXPECT_EQ(ctx.getTTIDs(), tts);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 2U);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 2U);
  EXPECT_TRUE(ctx.contains(*elfHip));

  EXPECT_TRUE(ctx.contains(*elfMulti));
}

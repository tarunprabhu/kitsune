//===- ArchiveUtilsTest.cpp - Tests for Kitsune's archive utilities -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Object/ArchiveUtils.h"
#include "CheckUtils.h"
#include "CompressedBinary.h"
#include "kitsune/Object/EmbDeviceCodeContext.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ObjectFile.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

TEST(ArchiveUtilsTest, hasEmbDeviceCode) {
  detail::check_false(hasEmbDeviceCode(*arEmpty));
  detail::check_false(hasEmbDeviceCode(*arNoDeviceCode));
  detail::check_true(hasEmbDeviceCode(*arCuda1));
  detail::check_true(hasEmbDeviceCode(*arHip1));
  detail::check_true(hasEmbDeviceCode(*arMulti));

  detail::check_err(hasEmbDeviceCode(*arHetero));
}

TEST(ArchiveUtilsTest, getEmbDeviceCodeTTIDs) {
  using Vec = SmallVector<TTID, 0>;

  detail::check_eq(getEmbDeviceCodeTTIDs(*arEmpty), Vec({}));
  detail::check_eq(getEmbDeviceCodeTTIDs(*arNoDeviceCode), Vec({}));
  detail::check_eq(getEmbDeviceCodeTTIDs(*arCuda1), Vec({TTID::Cuda}));
  detail::check_eq(getEmbDeviceCodeTTIDs(*arHip1), Vec({TTID::Hip}));

  Expected<SmallVector<TTID, 0>> tts = getEmbDeviceCodeTTIDs(*arMulti);
  EXPECT_TRUE(bool(tts));
  std::sort(tts->begin(), tts->end());
  EXPECT_EQ(*tts, Vec({TTID::Cuda, TTID::Hip}));

  detail::check_err(getEmbDeviceCodeTTIDs(*arHetero));
}

TEST(ArchiveUtilsTest, getNumMembers) {
  detail::check_eq(getNumMembers(*arEmpty), 0UL);
  detail::check_eq(getNumMembers(*arNoDeviceCode), 1UL);
  detail::check_eq(getNumMembers(*arCuda1), 1UL);
  detail::check_eq(getNumMembers(*arHip1), 1UL);
  detail::check_eq(getNumMembers(*arMulti), 2UL);
  detail::check_eq(getNumMembers(*arHetero), 2UL);
}

TEST(ArchiveUtilsTest, getMemberObjects) {
  auto check = [](const Archive &archive,
                  const SmallSet<StringRef, 2> &fileNames) {
    Expected<ArchiveMemberObjects> objs = getMemberObjects(archive);
    EXPECT_TRUE(bool(objs));

    EXPECT_EQ(objs->size(), fileNames.size());
    for (const std::unique_ptr<ObjectFile> &obj : *objs)
      EXPECT_TRUE(fileNames.contains(obj->getFileName()));
  };

  check(*arEmpty, {});
  check(*arNoDeviceCode, {"empty.o"});
  check(*arCuda1, {"sm_72.o"});
  check(*arHip1, {"gfx906.o"});
  check(*arMulti, {"sm_72.o", "gfx906.o"});

  detail::check_err(getMemberObjects(*arHetero));
}

// The add* tests actually test EmbDeviceCodeContext::add(ObjectFile). But these
// are tested here since the are defined in kitsune/lib/Object/ObjectUtils.cpp.
TEST(ArchiveUtilsTest, addEmpty) {
  EmbDeviceCodeContext ctx;

  Expected<unsigned> res = ctx.add(cast<Binary>(*elfEmpty));
  EXPECT_TRUE(bool(res));
  EXPECT_TRUE(*res == 0);
  EXPECT_TRUE(ctx.getTTIDs().empty());
  EXPECT_FALSE(ctx.contains(*arEmpty));
}

TEST(ArchiveUtilsTest, addNoDeviceCode) {
  EmbDeviceCodeContext ctx;

  Expected<unsigned> res = ctx.add(cast<Binary>(*arNoDeviceCode));
  EXPECT_TRUE(bool(res));
  EXPECT_TRUE(*res == 0);
  EXPECT_TRUE(ctx.getTTIDs().empty());
  EXPECT_FALSE(ctx.contains(*arNoDeviceCode));
}

TEST(ArchiveUtilsTest, addCuda1) {
  EmbDeviceCodeContext ctx;
  SmallVector<TTID, 2> tts = {TTID::Cuda};

  Expected<unsigned> res1 = ctx.add(cast<Binary>(*arCuda1));
  EXPECT_TRUE(bool(res1));
  EXPECT_EQ(*res1, 1U);
  EXPECT_EQ(ctx.getTTIDs(), tts);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 1UL);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 0UL);
  EXPECT_TRUE(ctx.contains(*arCuda1));

  Expected<unsigned> res2 = ctx.add(cast<Binary>(*arCuda1));
  EXPECT_TRUE(bool(res2));
  EXPECT_EQ(*res2, 0U);
  EXPECT_EQ(ctx.getTTIDs(), tts);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 1UL);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 0UL);
}

TEST(ArchiveUtilsTest, addHip) {
  EmbDeviceCodeContext ctx;
  SmallVector<TTID, 2> tts = {TTID::Hip};

  Expected<unsigned> res1 = ctx.add(cast<Binary>(*arHip1));
  EXPECT_TRUE(bool(res1));
  EXPECT_EQ(*res1, 1U);
  EXPECT_EQ(ctx.getTTIDs(), tts);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 0UL);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 1UL);
  EXPECT_TRUE(ctx.contains(*arHip1));

  Expected<unsigned> res2 = ctx.add(cast<Binary>(*arHip1));
  EXPECT_TRUE(bool(res2));
  EXPECT_EQ(*res2, 0U);
  EXPECT_EQ(ctx.getTTIDs(), tts);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 0UL);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 1UL);
}

TEST(ArchiveUtilsTest, addMulti) {
  EmbDeviceCodeContext ctx;
  SmallVector<TTID, 2> tts = {TTID::Cuda, TTID::Hip};

  Expected<unsigned> res1 = ctx.add(cast<Binary>(*arMulti));
  EXPECT_TRUE(bool(res1));
  EXPECT_EQ(*res1, 2U);
  EXPECT_EQ(ctx.getTTIDs(), tts);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 1U);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 1U);
  EXPECT_TRUE(ctx.contains(*arMulti));

  Expected<unsigned> res2 = ctx.add(cast<Binary>(*arCuda1));
  EXPECT_TRUE(bool(res2));
  EXPECT_EQ(*res2, 1U);
  EXPECT_EQ(ctx.getTTIDs(), tts);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 2U);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 1U);
  EXPECT_TRUE(ctx.contains(*arCuda1));

  Expected<unsigned> res3 = ctx.add(cast<Binary>(*arHip1));
  EXPECT_TRUE(bool(res3));
  EXPECT_EQ(*res3, 1U);
  EXPECT_EQ(ctx.getTTIDs(), tts);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 2U);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 2U);
  EXPECT_TRUE(ctx.contains(*arHip1));

  const SmallVectorImpl<EmbDeviceCode> &cudas = ctx.get(TTID::Cuda);
  EXPECT_EQ(cudas[0].getName(), "arMulti-sm_72.a");
  EXPECT_EQ(cudas[1].getName(), "arCuda1-sm_72.a");

  const SmallVectorImpl<EmbDeviceCode> &hips = ctx.get(TTID::Hip);
  EXPECT_EQ(hips[0].getName(), "arMulti-gfx906.a");
  EXPECT_EQ(hips[1].getName(), "arHip1-gfx906.a");

  EXPECT_TRUE(ctx.contains(*arMulti));
}

TEST(ArchiveUtilsTest, addHetero) {
  EmbDeviceCodeContext ctx;

  detail::check_err(ctx.add(cast<Binary>(*arHetero)));
  EXPECT_TRUE(ctx.getTTIDs().empty());
  EXPECT_FALSE(ctx.contains(*arHetero));
}

TEST(ArchiveUtilsTest, addMemBuf) {
  EmbDeviceCodeContext ctx;
  MemoryBufferRef memBuf = arCuda1->getMemoryBufferRef();

  Expected<unsigned> res1 = ctx.add(memBuf);
  EXPECT_TRUE(bool(res1));
  EXPECT_EQ(*res1, 1U);
  EXPECT_EQ(ctx.getTTIDs().size(), 1U);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 1U);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 0U);

  // When adding a memory buffer, we do not check the contents of the buffer, so
  // multiple buffers with identical contents can be added to the context.
  Expected<unsigned> res2 = ctx.add(memBuf);
  EXPECT_TRUE(bool(res2));
  EXPECT_EQ(*res2, 1U);
  EXPECT_EQ(ctx.getTTIDs().size(), 1U);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 2U);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 0U);
}

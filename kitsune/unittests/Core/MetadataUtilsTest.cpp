//===- MetadataUtilsTest.cpp - Unit tests for metadata utilities ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/MetadataUtils.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitMetadataUtils, fromMetadata) {
  std::string s = "courtauld institute of art";
  StringLiteral lit = "london school of hygiene and tropical medicine";

  LLVMContext ctx;
  Metadata *mi32 = toMetadata(1, ctx);
  Metadata *mi64 = toMetadata(1L, ctx);
  Metadata *mf32 = toMetadata(1.0F, ctx);
  Metadata *mf64 = toMetadata(1.0, ctx);
  Metadata *mTTID = toMetadata(TTID::Serial, ctx);
  Metadata *mStrat = toMetadata(TapirSpawnStrategy::Sequential, ctx);
  Metadata *mcstr = toMetadata("clare", ctx);
  Metadata *mstring = toMetadata(s, ctx);
  Metadata *mlit = toMetadata(lit, ctx);

  EXPECT_FALSE(fromMetadata<int32_t>(mi64));
  EXPECT_FALSE(fromMetadata<int32_t>(mf32));
  EXPECT_FALSE(fromMetadata<int64_t>(mi32));
  EXPECT_FALSE(fromMetadata<int64_t>(mf64));

  EXPECT_FALSE(fromMetadata<float>(mf64));
  EXPECT_FALSE(fromMetadata<float>(mi32));
  EXPECT_FALSE(fromMetadata<double>(mf32));
  EXPECT_FALSE(fromMetadata<double>(mi64));

  EXPECT_FALSE(fromMetadata<int32_t>(mcstr));
  EXPECT_FALSE(fromMetadata<int64_t>(mstring));
  EXPECT_FALSE(fromMetadata<double>(mTTID));
  EXPECT_FALSE(fromMetadata<StringRef>(mStrat));

  EXPECT_FALSE(fromMetadata<TTID>(mi64));
  EXPECT_FALSE(fromMetadata<TTID>(mf64));
  EXPECT_FALSE(fromMetadata<TTID>(mf32));
  EXPECT_FALSE(fromMetadata<TTID>(mlit));
}

TEST(KitMetadataUtils, toMetadata) {
  std::string s = "lucy cavendish";
  StringLiteral lit = "heriot-watt";

  LLVMContext ctx;
  Metadata *mi32 = toMetadata(1, ctx);
  Metadata *mi64 = toMetadata(1L, ctx);
  Metadata *mf32 = toMetadata(1.0F, ctx);
  Metadata *mf64 = toMetadata(1.0, ctx);
  Metadata *mTTID = toMetadata(TTID::Serial, ctx);
  Metadata *mStrat = toMetadata(TapirSpawnStrategy::Sequential, ctx);
  Metadata *mcstr = toMetadata("dundee", ctx);
  Metadata *mstring = toMetadata(s, ctx);
  Metadata *mlit = toMetadata(lit, ctx);

  EXPECT_TRUE(isa<ConstantAsMetadata>(mi32));
  EXPECT_TRUE(isa<ConstantAsMetadata>(mi64));
  EXPECT_TRUE(isa<ConstantAsMetadata>(mf32));
  EXPECT_TRUE(isa<ConstantAsMetadata>(mf64));
  EXPECT_TRUE(isa<ConstantAsMetadata>(mTTID));
  EXPECT_TRUE(isa<ConstantAsMetadata>(mStrat));
  EXPECT_TRUE(isa<MDString>(mcstr));
  EXPECT_TRUE(isa<MDString>(mstring));
  EXPECT_TRUE(isa<MDString>(mlit));

  EXPECT_EQ(fromMetadata<int32_t>(mi32), 1);
  EXPECT_EQ(fromMetadata<int64_t>(mi64), 1L);
  EXPECT_EQ(fromMetadata<float>(mf32), 1.0F);
  EXPECT_EQ(fromMetadata<double>(mf64), 1.0);
  EXPECT_EQ(fromMetadata<TTID>(mTTID), TTID::Serial);
  EXPECT_EQ(fromMetadata<TapirSpawnStrategy>(mStrat),
            TapirSpawnStrategy::Sequential);
  EXPECT_EQ(fromMetadata<StringRef>(mcstr), "dundee");
  EXPECT_EQ(fromMetadata<StringRef>(mstring), "lucy cavendish");
  EXPECT_EQ(fromMetadata<StringRef>(mlit), "heriot-watt");
}

TEST(KitMetadata, toFromVec) {
  LLVMContext ctx;
  SmallVector<int, 0> v;

  Metadata *md0 = toMetadata(v, ctx);
  EXPECT_TRUE(isa<MDNode>(md0));
  EXPECT_EQ(cast<MDNode>(md0)->getNumOperands(), 0U);
  EXPECT_EQ((fromMetadata<SmallVector<int, 0>>(md0)), v);

  v.push_back(11);
  Metadata *md1 = toMetadata(v, ctx);
  EXPECT_TRUE(isa<MDNode>(md1));
  EXPECT_EQ(cast<MDNode>(md1)->getNumOperands(), 1U);
  EXPECT_EQ((fromMetadata<SmallVector<int, 0>>(md1)), v);

  v.push_back(22);
  Metadata *md2 = toMetadata(v, ctx);
  EXPECT_TRUE(isa<MDNode>(md2));
  EXPECT_EQ(cast<MDNode>(md2)->getNumOperands(), 2U);
  EXPECT_EQ((fromMetadata<SmallVector<int, 0>>(md2)), v);
}

TEST(KitMetadata, toFromSet) {
  LLVMContext ctx;
  SmallSet<int, 0> v;

  Metadata *md0 = toMetadata(v, ctx);
  EXPECT_TRUE(isa<MDNode>(md0));
  EXPECT_EQ(cast<MDNode>(md0)->getNumOperands(), 0U);
  EXPECT_EQ((fromMetadata<SmallSet<int, 0>>(md0)), v);

  v.insert(11);
  Metadata *md1 = toMetadata(v, ctx);
  EXPECT_TRUE(isa<MDNode>(md1));
  EXPECT_EQ(cast<MDNode>(md1)->getNumOperands(), 1U);
  EXPECT_EQ((fromMetadata<SmallSet<int, 0>>(md1)), v);

  v.insert(22);
  Metadata *md2 = toMetadata(v, ctx);
  EXPECT_TRUE(isa<MDNode>(md2));
  EXPECT_EQ(cast<MDNode>(md2)->getNumOperands(), 2U);
  EXPECT_EQ((fromMetadata<SmallSet<int, 0>>(md2)), v);
}

} // namespace

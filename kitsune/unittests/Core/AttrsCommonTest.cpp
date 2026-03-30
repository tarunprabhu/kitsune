//===- AttrsCommonTest.cpp - Unit tests for Kitsune's attr utilities ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/AttrsCommon.h"
#include "Core/AttrsImpl.h"
#include "llvm/IR/Metadata.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

static constexpr StringRef attrOld = "roadrunner";
static constexpr StringRef attrNew = "biscohitos";

static void checkAttrList(MDNode *md, unsigned expNumAttrs) {
  EXPECT_TRUE(md->isDistinct());
  EXPECT_EQ(md->getNumOperands(), expNumAttrs + 1);
  EXPECT_EQ(md->getOperand(0), md);
  for (unsigned i = 1; i <= expNumAttrs; ++i)
    EXPECT_TRUE(isa<MDNode>(md->getOperand(i)));
}

static void checkAttrFlag(Metadata *mdop) {
  EXPECT_TRUE(isa<MDNode>(mdop));

  MDNode *md = cast<MDNode>(mdop);
  EXPECT_EQ(md->getNumOperands(), 1U);

  Metadata *md0 = md->getOperand(0);
  EXPECT_TRUE(isa<MDString>(md0));
  EXPECT_EQ(cast<MDString>(md0)->getString(), "attr-flag");
}

static void checkAttr(Metadata *mdop, StringRef expVal) {
  EXPECT_TRUE(isa<MDNode>(mdop));

  MDNode *md = cast<MDNode>(mdop);
  EXPECT_EQ(md->getNumOperands(), 2U);

  Metadata *md0 = md->getOperand(0);
  Metadata *md1 = md->getOperand(1);
  EXPECT_TRUE(isa<MDString>(md0));
  EXPECT_TRUE(isa<MDString>(md1));
  EXPECT_EQ(cast<MDString>(md0)->getString(), "attr-new");
  EXPECT_EQ(cast<MDString>(md1)->getString(), expVal);
}

TEST(KitAttrsCommon, makeRawAttr) {
  LLVMContext ctx;
  MDNode *md = nullptr;
  Metadata *strOld = MDString::get(ctx, attrOld);
  Metadata *strNew = MDString::get(ctx, attrNew);

  md = detail::makeRawAttr(ctx, "attr-flag", {});
  checkAttrFlag(md);

  md = detail::makeRawAttr(ctx, "attr-new", {strOld});
  checkAttr(md, attrOld);

  md = detail::makeRawAttr(ctx, "attr2", {strOld, strNew});
  EXPECT_EQ(cast<MDString>(md->getOperand(0))->getString(), "attr2");
  EXPECT_EQ(cast<MDString>(md->getOperand(1))->getString(), attrOld);
  EXPECT_EQ(cast<MDString>(md->getOperand(2))->getString(), attrNew);
}

TEST(KitAttrsCommon, newAttrList) {
  LLVMContext ctx;
  MDNode *md = nullptr;

  md = detail::getNewAttrList(ctx);
  checkAttrList(md, 0);
}

TEST(KitAttrsCommon, newAttrListWith) {
  LLVMContext ctx;
  MDNode *md = nullptr;

  md = getAttrListWith("attr-new", {MDString::get(ctx, attrOld)}, nullptr, ctx);
  checkAttrList(md, 1);
  checkAttr(md->getOperand(1), attrOld);

  md = getAttrListWith("attr-flag", {}, md, ctx);
  checkAttrList(md, 2);
  checkAttr(md->getOperand(1), attrOld);
  checkAttrFlag(md->getOperand(2));

  md = getAttrListWith("attr-new", {MDString::get(ctx, attrNew)}, md, ctx);
  checkAttrList(md, 2);
  checkAttr(md->getOperand(1), attrNew);
  checkAttrFlag(md->getOperand(2));

  md = detail::getNewAttrList(ctx);
  md = getAttrListWith("attr-flag", {}, md, ctx);
  checkAttrList(md, 1);
  checkAttrFlag(md->getOperand(1));
}

TEST(KitAttrsCommon, newAttrListWithout) {
  LLVMContext ctx;
  MDNode *md = nullptr;
  MDNode *mdEmpty = nullptr;

  EXPECT_FALSE(getAttrListWithout("attr-flag", nullptr));

  mdEmpty = detail::getNewAttrList(ctx);
  checkAttrList(mdEmpty, 0);

  md = getAttrListWithout("attr-flag", mdEmpty);
  EXPECT_FALSE(md);

  md = getAttrListWith("attr-flag", {}, md, ctx);
  checkAttrList(md, 1);
  checkAttrFlag(md->getOperand(1));

  md = getAttrListWithout("attr-flag", md);
  EXPECT_FALSE(md);

  md = getAttrListWith("attr-new", {MDString::get(ctx, attrOld)}, md, ctx);
  md = getAttrListWith("attr-flag", {}, md, ctx);
  md = getAttrListWithout("attr-new", md);
  checkAttrList(md, 1);
  checkAttrFlag(md->getOperand(1));
}

TEST(KitAttrsCommon, getRawAttr) {
  LLVMContext ctx;
  MDNode *md = nullptr;
  MDNode *mdAttr = nullptr;

  md = detail::getNewAttrList(ctx);
  EXPECT_FALSE(detail::getRawAttr("attr-flag", md));

  md = getAttrListWith("attr-flag", {}, md, ctx);
  mdAttr = detail::getRawAttr("attr-flag", md);
  EXPECT_TRUE(mdAttr);
  checkAttrFlag(mdAttr);

  EXPECT_FALSE(detail::getRawAttr("attr-new", md));
}

TEST(KitAttrsCommon, getRawAttrValue) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  Metadata *v0 = MDString::get(ctx, "beeblebrox");
  Metadata *v1 = ConstantAsMetadata::get(ConstantInt::get(i32, 11011010));
  MDNode *md = nullptr;
  MDNode *attr = nullptr;

  md = getAttrListWith("attr", {v0, v1}, nullptr, ctx);
  attr = detail::getRawAttr("attr", md);

  // Out of range.
  EXPECT_FALSE(detail::getRawAttrValue<StringRef>(*attr, 2));
  EXPECT_FALSE(detail::getRawAttrValue<StringRef>(*attr, 3));

  // Incorrect type at index.
  EXPECT_FALSE(detail::getRawAttrValue<int32_t>(*attr, 0));
  EXPECT_FALSE(detail::getRawAttrValue<int64_t>(*attr, 1));

  // Correct type at index.
  EXPECT_EQ(detail::getRawAttrValue<StringRef>(*attr, 0), "beeblebrox");
  EXPECT_EQ(detail::getRawAttrValue<int32_t>(*attr, 1), 11011010);
}

} // namespace

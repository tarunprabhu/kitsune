//===- SingletonUtilsTest.cpp - Unit tests for embedded data utilities ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/SingletonUtils.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

TEST(KitsuneSingletonUtils, getTTIDFromSection) {
  EXPECT_EQ(getTTIDForSection(KITSUNE_CUDA_CODE_SECTION), TTID::Cuda);
  EXPECT_EQ(getTTIDForSection(KITSUNE_HIP_CODE_SECTION), TTID::Hip);

  EXPECT_EQ(getTTIDForSection(""), std::nullopt);
  EXPECT_EQ(getTTIDForSection(KITSUNE_CUDA_FB_SECTION), std::nullopt);
  EXPECT_EQ(getTTIDForSection(KITSUNE_HIP_FB_SECTION), std::nullopt);
}

TEST(KitsuneSingletonUtils, getName) {
  EXPECT_EQ(getSingletonFBName(TTID::Cuda), KITSUNE_CUDA_FB_NAME);
  EXPECT_EQ(getSingletonFBName(TTID::Hip), KITSUNE_HIP_FB_NAME);
}

TEST(KitsuneSingletonUtils, getSection) {
  EXPECT_EQ(getSingletonFBSection(TTID::Cuda), KITSUNE_CUDA_FB_SECTION);
  EXPECT_EQ(getSingletonFBSection(TTID::Hip), KITSUNE_HIP_FB_SECTION);
}

TEST(KitEmbUtils, createSingletonFBCuda) {
  LLVMContext ctx;
  Module m("", ctx);

  EXPECT_FALSE(getSingletonFBGlobal(TTID::Cuda, m));

  GlobalVariable *g = createSingletonFBGlobal(TTID::Cuda, m);

  EXPECT_TRUE(g->hasName());
  EXPECT_EQ(g->getName(), KITSUNE_CUDA_FB_NAME);
  EXPECT_TRUE(g->isConstant());
  EXPECT_FALSE(g->hasInitializer());
  EXPECT_TRUE(isa<ArrayType>(g->getValueType()));
  EXPECT_EQ(cast<ArrayType>(g->getValueType())->getNumElements(), 0U);
  EXPECT_EQ(g->getSection(), KITSUNE_CUDA_FB_SECTION);
  EXPECT_TRUE(g->hasAttribute(Attribute::KitFB));
  EXPECT_TRUE(g->hasAttribute(Attribute::KitTT));
  EXPECT_EQ(g->getAttribute(Attribute::KitTT).getTTID(), TTID::Cuda);

  EXPECT_EQ(g->getParent(), &m);
  EXPECT_EQ(getSingletonFBGlobal(TTID::Cuda, m), g);
}

TEST(KitEmbUtils, createEmbFBHip) {
  LLVMContext ctx;
  Module m("", ctx);

  EXPECT_FALSE(getSingletonFBGlobal(TTID::Hip, m));

  GlobalVariable *g = createSingletonFBGlobal(TTID::Hip, m);

  EXPECT_TRUE(g->hasName());
  EXPECT_EQ(g->getName(), KITSUNE_HIP_FB_NAME);
  EXPECT_TRUE(g->isConstant());
  EXPECT_FALSE(g->hasInitializer());
  EXPECT_TRUE(isa<ArrayType>(g->getValueType()));
  EXPECT_EQ(cast<ArrayType>(g->getValueType())->getNumElements(), 0U);
  EXPECT_EQ(g->getSection(), KITSUNE_HIP_FB_SECTION);
  EXPECT_EQ(g->getUnnamedAddr(), GlobalValue::UnnamedAddr::None);
  EXPECT_TRUE(g->hasAttribute(Attribute::KitFB));
  EXPECT_TRUE(g->hasAttribute(Attribute::KitTT));
  EXPECT_EQ(g->getAttribute(Attribute::KitTT).getTTID(), TTID::Hip);

  EXPECT_EQ(g->getParent(), &m);
  EXPECT_EQ(getSingletonFBGlobal(TTID::Hip, m), g);
}

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

TEST(KitsuneSingletonUtils, names) {
  EXPECT_EQ(cudaFatbinName, "__nv_fatbin");
  EXPECT_EQ(cudaFatbinSection, ".nv_fatbin");
  EXPECT_EQ(cudaBundleSection, ".nvFatBinSegment");

  EXPECT_EQ(hipFatbinName, "__hip_fatbin");
  EXPECT_EQ(hipFatbinSection, ".hip_fatbin");
  EXPECT_EQ(hipBundleSection, ".hipFatBinSegment");
}

TEST(KitSingletonUtils, getName) {
  EXPECT_EQ(getFatbinName(TTID::Cuda), cudaFatbinName);
  EXPECT_EQ(getFatbinName(TTID::Hip), hipFatbinName);
}

TEST(KitsuneSingletonUtils, getSection) {
  EXPECT_EQ(getFatbinSection(TTID::Cuda), cudaFatbinSection);
  EXPECT_EQ(getFatbinSection(TTID::Hip), hipFatbinSection);
}

TEST(KitEmbUtils, createSingletonFBCuda) {
  LLVMContext ctx;
  Module m("", ctx);

  EXPECT_FALSE(getSingletonFBGlobal(TTID::Cuda, m));

  GlobalVariable *g = createSingletonFBGlobal(TTID::Cuda, m);

  EXPECT_TRUE(g->hasName());
  EXPECT_EQ(g->getName(), cudaFatbinName);
  EXPECT_TRUE(g->isConstant());
  EXPECT_FALSE(g->hasInitializer());
  EXPECT_TRUE(isa<ArrayType>(g->getValueType()));
  EXPECT_EQ(cast<ArrayType>(g->getValueType())->getNumElements(), 0U);
  EXPECT_EQ(g->getSection(), cudaFatbinSection);
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
  EXPECT_EQ(g->getName(), hipFatbinName);
  EXPECT_TRUE(g->isConstant());
  EXPECT_FALSE(g->hasInitializer());
  EXPECT_TRUE(isa<ArrayType>(g->getValueType()));
  EXPECT_EQ(cast<ArrayType>(g->getValueType())->getNumElements(), 0U);
  EXPECT_EQ(g->getSection(), hipFatbinSection);
  EXPECT_EQ(g->getUnnamedAddr(), GlobalValue::UnnamedAddr::None);
  EXPECT_TRUE(g->hasAttribute(Attribute::KitFB));
  EXPECT_TRUE(g->hasAttribute(Attribute::KitTT));
  EXPECT_EQ(g->getAttribute(Attribute::KitTT).getTTID(), TTID::Hip);

  EXPECT_EQ(g->getParent(), &m);
  EXPECT_EQ(getSingletonFBGlobal(TTID::Hip, m), g);
}

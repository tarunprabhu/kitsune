//===- EmbUtilsTest.cpp - Unit tests for embedded data utilities ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/EmbUtils.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

static constexpr StringRef fcuda = R"(
define i32 @fcuda(i32 %n) {
entry:
  ret i32 %n
}
)";

static constexpr StringRef fhip = R"(
define i32 @fhip(i32 %n) {
entry:
  ret i32 %n
}
)";

static std::unique_ptr<Module> parseIR(LLVMContext &ctx, StringRef ir) {
  SMDiagnostic err;
  std::unique_ptr<Module> m = parseAssemblyString(ir, err, ctx);
  if (!m)
    err.print("parseIR", errs());
  return m;
}

TEST(KitEmbUtils, createEmbBCGlobal) {
  LLVMContext ctx;
  std::unique_ptr<Module> hostM = parseIR(ctx, R"()");

  std::unique_ptr<Module> cudaM = parseIR(ctx, fcuda);
  GlobalVariable *gc = createEmbBCGlobal(*cudaM, TTID::Cuda, *hostM);
  EXPECT_TRUE(gc->hasName());
  EXPECT_TRUE(gc->hasAttribute(Attribute::KitBC));
  EXPECT_TRUE(gc->hasAttribute(Attribute::KitTT));
  EXPECT_EQ(gc->getAttribute(Attribute::KitTT).getTTID(), TTID::Cuda);
  EXPECT_EQ(gc->getParent(), hostM.get());
  EXPECT_EQ(getEmbBCGlobal(TTID::Cuda, *hostM), gc);

  std::unique_ptr<Module> hipM = parseIR(ctx, fhip);
  GlobalVariable *gh = createEmbBCGlobal(*hipM, TTID::Hip, *hostM);
  EXPECT_TRUE(gh->hasName());
  EXPECT_TRUE(gh->hasAttribute(Attribute::KitBC));
  EXPECT_TRUE(gh->hasAttribute(Attribute::KitTT));
  EXPECT_EQ(gh->getAttribute(Attribute::KitTT).getTTID(), TTID::Hip);
  EXPECT_EQ(gh->getParent(), hostM.get());
  EXPECT_EQ(getEmbBCGlobal(TTID::Hip, *hostM), gh);
}

TEST(KitEmbUtils, parseEmbBCGlobal) {
  LLVMContext ctx;
  std::unique_ptr<Module> hostM = parseIR(ctx, R"()");
  std::unique_ptr<Module> cudaM = parseIR(ctx, fcuda);

  GlobalVariable *gCuda = createEmbBCGlobal(*cudaM, TTID::Cuda, *hostM);
  Expected<std::unique_ptr<Module>> parseCudaMOrErr = parseEmbBCGlobal(*gCuda);
  EXPECT_EQ((bool)parseCudaMOrErr, true);

  std::unique_ptr<Module> parseCudaM = std::move(*parseCudaMOrErr);
  EXPECT_TRUE(parseCudaM->getFunction("fcuda"));
  EXPECT_EQ(parseCudaM->getName(), "");

  std::unique_ptr<Module> hipM = parseIR(ctx, R"(
define i32 @fhip(i32 %n) {
entry:
  ret i32 %n
}

!kitsune.device.module.flags = !{!0, !1}

!0 = !{i32 4}
!1 = !{!"some-silly-name"}
)");

  GlobalVariable *gHip = createEmbBCGlobal(*hipM, TTID::Hip, *hostM);
  Expected<std::unique_ptr<Module>> parseHipMOrErr = parseEmbBCGlobal(*gHip);
  EXPECT_EQ((bool)parseHipMOrErr, true);

  std::unique_ptr<Module> parseHipM = std::move(*parseHipMOrErr);
  EXPECT_TRUE(parseHipM->getFunction("fhip"));
  EXPECT_EQ(parseHipM->getName(), "some-silly-name");
}

TEST(KitEmbUtils, resetEmbBCGlobal) {
  LLVMContext ctx;
  std::unique_ptr<Module> parseM = nullptr;
  std::unique_ptr<Module> hostM = parseIR(ctx, R"()");
  std::unique_ptr<Module> cudaM = parseIR(ctx, fcuda);
  std::unique_ptr<Module> hipM = parseIR(ctx, fhip);

  GlobalVariable *g0 = createEmbBCGlobal(*cudaM, TTID::Cuda, *hostM);
  g0->setName("g0");

  Expected<std::unique_ptr<Module>> parseMOrErr0 = parseEmbBCGlobal(*g0);
  EXPECT_EQ((bool)parseMOrErr0, true);

  parseM = std::move(*parseMOrErr0);
  EXPECT_TRUE(hostM->getGlobalVariable("g0"));
  EXPECT_FALSE(hostM->getGlobalVariable("g1"));
  EXPECT_TRUE(parseM->getFunction("fcuda"));
  EXPECT_FALSE(parseM->getFunction("fhip"));

  GlobalVariable *g1 = resetEmbBCGlobal(*hipM, *g0);

  Expected<std::unique_ptr<Module>> parseMOrErr1 = parseEmbBCGlobal(*g1);
  EXPECT_EQ((bool)parseMOrErr1, true);

  parseM = std::move(*parseMOrErr1);
  EXPECT_EQ(g1->getName(), "g0");
  EXPECT_TRUE(hostM->getGlobalVariable("g0"));
  EXPECT_EQ(hostM->global_size(), 1U);
  EXPECT_FALSE(parseM->getFunction("fcuda"));
  EXPECT_TRUE(parseM->getFunction("fhip"));
  EXPECT_EQ(g1->getAttribute(Attribute::KitTT).getTTID(), TTID::Cuda);
}

TEST(KitEmbUtils, getEmbModule) {
  LLVMContext ctx;
  std::unique_ptr<Module> hostM = parseIR(ctx, R"()");
  std::unique_ptr<Module> cudaM = parseIR(ctx, fcuda);
  std::unique_ptr<Module> hipM = parseIR(ctx, fhip);

  {
    Expected<std::unique_ptr<Module>> embM = getEmbModule(TTID::Cuda, *hostM);
    EXPECT_EQ((bool)embM, true);
    EXPECT_FALSE(*embM);
  }

  {
    createEmbBCGlobal(*cudaM, TTID::Cuda, *hostM);
    Expected<std::unique_ptr<Module>> embM = getEmbModule(TTID::Cuda, *hostM);
    EXPECT_EQ((bool)embM, true);
    EXPECT_EQ((*embM)->size(), 1U);
    EXPECT_TRUE((*embM)->getFunction("fcuda"));
  }

  {
    Expected<std::unique_ptr<Module>> embM = getEmbModule(TTID::Hip, *hostM);
    EXPECT_EQ((bool)embM, true);
    EXPECT_FALSE(*embM);
  }
}

TEST(KitEmbUtils, getEmbModules) {
  LLVMContext ctx;
  std::unique_ptr<Module> hostM = parseIR(ctx, R"()");
  std::unique_ptr<Module> cudaM = parseIR(ctx, fcuda);
  std::unique_ptr<Module> hipM = parseIR(ctx, fhip);

  {
    Expected<EmbModulesMapTy> embMs = getEmbModules(*hostM);
    EXPECT_EQ((bool)embMs, true);
    EXPECT_EQ(embMs->size(), 0U);
  }

  {
    createEmbBCGlobal(*cudaM, TTID::Cuda, *hostM);
    Expected<EmbModulesMapTy> embMs = getEmbModules(*hostM);
    EXPECT_EQ((bool)embMs, true);
    EXPECT_EQ(embMs->size(), 1U);
    EXPECT_TRUE((*embMs)[TTID::Cuda]->getFunction("fcuda"));
  }

  {
    createEmbBCGlobal(*hipM, TTID::Hip, *hostM);
    Expected<EmbModulesMapTy> embMs = getEmbModules(*hostM);
    EXPECT_EQ((bool)embMs, true);
    EXPECT_EQ(embMs->size(), 2U);
    EXPECT_TRUE((*embMs)[TTID::Cuda]->getFunction("fcuda"));
    EXPECT_TRUE((*embMs)[TTID::Hip]->getFunction("fhip"));
  }
}

TEST(KitEmbUtils, createEmbFBCuda) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, R"()");

  EXPECT_FALSE(getEmbFBGlobal(TTID::Cuda, *m));

  GlobalVariable *g = createEmbFBGlobal(TTID::Cuda, *m);

  EXPECT_TRUE(g->hasInitializer());
  EXPECT_TRUE(g->isConstant());
  EXPECT_TRUE(g->getInitializer()->isZeroValue());
  EXPECT_TRUE(isa<ArrayType>(g->getValueType()));
  EXPECT_EQ(cast<ArrayType>(g->getValueType())->getNumElements(), 0U);
  EXPECT_EQ(g->getSection(), ".nv_fatbin");
  EXPECT_TRUE(g->hasAttribute(Attribute::KitFB));
  EXPECT_TRUE(g->hasAttribute(Attribute::KitTT));
  EXPECT_EQ(g->getAttribute(Attribute::KitTT).getTTID(), TTID::Cuda);
  EXPECT_EQ(g->getParent(), m.get());
  EXPECT_EQ(getEmbFBGlobal(TTID::Cuda, *m), g);
}

TEST(KitEmbUtils, createEmbFBHip) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, R"()");

  EXPECT_FALSE(getEmbFBGlobal(TTID::Hip, *m));

  GlobalVariable *g = createEmbFBGlobal(TTID::Hip, *m);

  EXPECT_TRUE(g->hasInitializer());
  EXPECT_TRUE(g->isConstant());
  EXPECT_TRUE(g->getInitializer()->isZeroValue());
  EXPECT_TRUE(isa<ArrayType>(g->getValueType()));
  EXPECT_EQ(cast<ArrayType>(g->getValueType())->getNumElements(), 0U);
  EXPECT_EQ(g->getSection(), ".hip_fatbin");
  EXPECT_EQ(g->getAlign(), Align(4096));
  EXPECT_EQ(g->getUnnamedAddr(), GlobalValue::UnnamedAddr::None);
  EXPECT_TRUE(g->hasAttribute(Attribute::KitFB));
  EXPECT_EQ(g->getAttribute(Attribute::KitTT).getTTID(), TTID::Hip);
  EXPECT_EQ(g->getParent(), m.get());
  EXPECT_EQ(getEmbFBGlobal(TTID::Hip, *m), g);
}

TEST(KitEmbUtils, resetEmbFB) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, R"()");
  std::unique_ptr<MemoryBuffer> buf = MemoryBuffer::getMemBuffer("repl");

  GlobalVariable *g0 = createEmbFBGlobal(TTID::Cuda, *m);
  g0->setName("g0");

  GlobalVariable *g1 = resetEmbFBGlobal(*buf, *g0);

  EXPECT_TRUE(m->getGlobalVariable("g0"));
  EXPECT_EQ(g1->getName(), "g0");
  EXPECT_EQ(m->global_size(), 1U);
  EXPECT_EQ(g1->getAttribute(Attribute::KitTT).getTTID(), TTID::Cuda);
  EXPECT_TRUE(isa<ConstantDataArray>(g1->getInitializer()));
  EXPECT_EQ(cast<ConstantDataArray>(g1->getInitializer())->getAsString(),
            "repl");
}

} // namespace

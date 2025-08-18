//==- EmbBitcodeUtilsTest.cpp - Unit tests for embedded bitcode utilities --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/EmbBitcodeUtils.h"
#include "kitsune/Core/SingletonUtils.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

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
  std::unique_ptr<Module> cudaM = parseIR(ctx, R"(
define i32 @fcuda(i32 %n) {
entry:
  ret i32 %n
}
)");

  GlobalVariable *g = createEmbBCGlobal(*cudaM, TTID::Cuda, *hostM);

  // The global variable containing embedded bitcode must have a name, otherwise
  // it may be deleted by the global variable optimizastion pass.
  EXPECT_TRUE(g->hasName());
  EXPECT_TRUE(g->hasAttribute(Attribute::KitBC));
  EXPECT_TRUE(g->hasAttribute(Attribute::KitTT));
  EXPECT_EQ(g->getAttribute(Attribute::KitTT).getTTID(), TTID::Cuda);
  EXPECT_EQ(g->getParent(), hostM.get());
  EXPECT_EQ(getEmbBCGlobal(TTID::Cuda, *hostM), g);
}

TEST(KitEmbUtils, parseEmbBCGlobal) {
  LLVMContext ctx;
  std::unique_ptr<Module> hostM = parseIR(ctx, R"()");
  std::unique_ptr<Module> cudaM = parseIR(ctx, R"(
define i32 @fcuda(i32 %n) {
entry:
  ret i32 %n
}
)");

  GlobalVariable *gCuda = createEmbBCGlobal(*cudaM, TTID::Cuda, *hostM);
  std::unique_ptr<Module> parseCudaM = parseEmbBCGlobal(*gCuda);

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
  std::unique_ptr<Module> parseHipM = parseEmbBCGlobal(*gHip);

  EXPECT_TRUE(parseHipM->getFunction("fhip"));
  EXPECT_EQ(parseHipM->getName(), "some-silly-name");
}

TEST(KitEmbUtils, resetEmbBCGlobal) {
  LLVMContext ctx;
  std::unique_ptr<Module> parseM = nullptr;
  std::unique_ptr<Module> hostM = parseIR(ctx, R"()");
  std::unique_ptr<Module> cudaM = parseIR(ctx, R"(
define i32 @fcuda(i32 %n) {
entry:
  ret i32 %n
}
)");
  std::unique_ptr<Module> hipM = parseIR(ctx, R"(
define i32 @fhip(i32 %n) {
entry:
  ret i32 %n
}
)");

  GlobalVariable *g0 = createEmbBCGlobal(*cudaM, TTID::Cuda, *hostM);
  g0->setName("g0");

  parseM = parseEmbBCGlobal(*g0);
  EXPECT_TRUE(hostM->getGlobalVariable("g0"));
  EXPECT_FALSE(hostM->getGlobalVariable("g1"));
  EXPECT_TRUE(parseM->getFunction("fcuda"));
  EXPECT_FALSE(parseM->getFunction("fhip"));

  GlobalVariable *g1 = resetEmbBCGlobal(*hipM, *g0);
  parseM = parseEmbBCGlobal(*g1);

  EXPECT_EQ(g1->getName(), "g0");
  EXPECT_TRUE(hostM->getGlobalVariable("g0"));
  EXPECT_EQ(hostM->global_size(), 1U);
  EXPECT_FALSE(parseM->getFunction("fcuda"));
  EXPECT_TRUE(parseM->getFunction("fhip"));
  EXPECT_EQ(g1->getAttribute(Attribute::KitTT).getTTID(), TTID::Cuda);
}

TEST(KitEmbUtils, getEmbModules) {
  LLVMContext ctx;
  std::unique_ptr<Module> cudaM = parseIR(ctx, R"(
define i32 @fcuda(i32 %n) {
entry:
  ret i32 %n
}
)");
  std::unique_ptr<Module> hipM = parseIR(ctx, R"(
define i32 @fhip(i32 %n) {
entry:
  ret i32 %n
}
)");
  std::unique_ptr<Module> hostM = parseIR(ctx, R"()");

  {
    EmbModulesMapTy embMs = getEmbModules(*hostM);
    EXPECT_EQ(embMs.size(), 0U);
  }

  {
    createEmbBCGlobal(*cudaM, TTID::Cuda, *hostM);
    EmbModulesMapTy embMs = getEmbModules(*hostM);
    EXPECT_EQ(embMs.size(), 1U);
    EXPECT_TRUE(embMs[TTID::Cuda]->getFunction("fcuda"));
  }

  {
    createEmbBCGlobal(*hipM, TTID::Hip, *hostM);
    EmbModulesMapTy embMs = getEmbModules(*hostM);
    EXPECT_EQ(embMs.size(), 2U);
    EXPECT_TRUE(embMs[TTID::Cuda]->getFunction("fcuda"));
    EXPECT_TRUE(embMs[TTID::Hip]->getFunction("fhip"));
  }
}

} // namespace

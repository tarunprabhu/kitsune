//===- KitsuneUtilsTest.cpp - Unit tests for kitsune's utilities ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/KitsuneUtils.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/KitsuneMetadata.h"
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

TEST(KitsuneUtils, createEmbeddedBC) {
  LLVMContext ctx;
  std::unique_ptr<Module> hostM = parseIR(ctx, R"()");
  std::unique_ptr<Module> cudaM = parseIR(ctx, R"(
define i32 @fcuda(i32 %n) {
entry:
  ret i32 %n
}
)");

  GlobalVariable *g = createEmbeddedBC(*cudaM, TapirTargetID::Cuda, *hostM);

  // The global variable containing embedded bitcode must have a name, otherwise
  // it may be deleted by the global variable optimizastion pass.
  EXPECT_TRUE(g->hasName());
  EXPECT_TRUE(hasKitsuneBCMD(*g, TapirTargetID::Cuda));
  EXPECT_FALSE(hasKitsuneBCMD(*g, TapirTargetID::Hip));
  EXPECT_EQ(g->getParent(), hostM.get());
  EXPECT_EQ(getEmbeddedBC(TapirTargetID::Cuda, *hostM), g);
}

TEST(KitsuneUtils, parseEmbeddedBC) {
  LLVMContext ctx;
  std::unique_ptr<Module> hostM = parseIR(ctx, R"()");
  std::unique_ptr<Module> cudaM = parseIR(ctx, R"(
define i32 @fcuda(i32 %n) {
entry:
  ret i32 %n
}
)");

  GlobalVariable *gCuda = createEmbeddedBC(*cudaM, TapirTargetID::Cuda, *hostM);
  std::unique_ptr<Module> parseCudaM = parseEmbeddedBC(*gCuda);

  EXPECT_TRUE(parseCudaM->getFunction("fcuda"));
  EXPECT_EQ(parseCudaM->getName(), "");

  std::unique_ptr<Module> hipM = parseIR(ctx, R"(
define i32 @fhip(i32 %n) {
entry:
  ret i32 %n
}

!kitsune.module.flags = !{!0, !1}

!0 = !{i8 3}
!1 = !{!"some-silly-name"}
)");

  GlobalVariable *gHip = createEmbeddedBC(*hipM, TapirTargetID::Hip, *hostM);
  std::unique_ptr<Module> parseHipM = parseEmbeddedBC(*gHip);

  EXPECT_TRUE(parseHipM->getFunction("fhip"));
  EXPECT_EQ(parseHipM->getName(), "some-silly-name");
}

TEST(KitsuneUtils, resetEmbeddedBC) {
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

  GlobalVariable *g0 = createEmbeddedBC(*cudaM, TapirTargetID::Cuda, *hostM);
  g0->setName("g0");
  g0->setMetadata("some.kitsune.md", MDNode::get(ctx, {}));

  parseM = parseEmbeddedBC(*g0);
  EXPECT_TRUE(hostM->getGlobalVariable("g0"));
  EXPECT_FALSE(hostM->getGlobalVariable("g1"));
  EXPECT_TRUE(parseM->getFunction("fcuda"));
  EXPECT_FALSE(parseM->getFunction("fhip"));

  GlobalVariable *g1 = resetEmbeddedBC(*hipM, *g0);
  g1->setName("g1");

  parseM = parseEmbeddedBC(*g1);
  EXPECT_TRUE(g1->hasMetadata("some.kitsune.md"));
  EXPECT_FALSE(hostM->getGlobalVariable("g0"));
  EXPECT_TRUE(hostM->getGlobalVariable("g1"));
  EXPECT_FALSE(parseM->getFunction("fcuda"));
  EXPECT_TRUE(parseM->getFunction("fhip"));
  EXPECT_TRUE(hasKitsuneBCMD(*g1, TapirTargetID::Cuda));
}

TEST(KitsuneUtils, getEmbeddedModules) {
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
    EmbeddedModulesMapTy embMs = getEmbeddedModules(*hostM);
    EXPECT_EQ(embMs.size(), 0U);
  }

  {
    createEmbeddedBC(*cudaM, TapirTargetID::Cuda, *hostM);
    EmbeddedModulesMapTy embMs = getEmbeddedModules(*hostM);
    EXPECT_EQ(embMs.size(), 1U);
    EXPECT_TRUE(embMs[TapirTargetID::Cuda]->getFunction("fcuda"));
  }

  {
    createEmbeddedBC(*hipM, TapirTargetID::Hip, *hostM);
    EmbeddedModulesMapTy embMs = getEmbeddedModules(*hostM);
    EXPECT_EQ(embMs.size(), 2U);
    EXPECT_TRUE(embMs[TapirTargetID::Cuda]->getFunction("fcuda"));
    EXPECT_TRUE(embMs[TapirTargetID::Hip]->getFunction("fhip"));
  }
}

TEST(KitsuneUtils, createEmbeddedFBCuda) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, R"()");

  EXPECT_FALSE(getEmbeddedFB(TapirTargetID::Cuda, *m));

  GlobalVariable *g = createEmbeddedFB(TapirTargetID::Cuda, *m);

  EXPECT_EQ(g->getSection(), ".nv_fatbin");
  EXPECT_TRUE(hasKitsuneFBMD(*g, TapirTargetID::Cuda));
  EXPECT_FALSE(hasKitsuneFBMD(*g, TapirTargetID::Hip));
  EXPECT_EQ(g->getParent(), m.get());
  EXPECT_EQ(getEmbeddedFB(TapirTargetID::Cuda, *m), g);
}

TEST(KitsuneUtils, createEmbeddedFBHip) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, R"()");

  EXPECT_FALSE(getEmbeddedFB(TapirTargetID::Hip, *m));

  GlobalVariable *g = createEmbeddedFB(TapirTargetID::Hip, *m);

  EXPECT_EQ(g->getSection(), ".hip_fatbin");
  EXPECT_EQ(g->getAlign(), Align(4096));
  EXPECT_EQ(g->getUnnamedAddr(), GlobalValue::UnnamedAddr::None);
  EXPECT_TRUE(hasKitsuneFBMD(*g, TapirTargetID::Hip));
  EXPECT_FALSE(hasKitsuneFBMD(*g, TapirTargetID::Cuda));
  EXPECT_EQ(g->getParent(), m.get());
  EXPECT_EQ(getEmbeddedFB(TapirTargetID::Hip, *m), g);
}

TEST(KitsuneUtils, resetEmbeddedFB) {
  auto getNumElements = [](Constant *c) -> unsigned {
    if (auto *cda = dyn_cast<ConstantDataArray>(c))
      return cda->getType()->getNumElements();
    return 0;
  };

  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, R"()");
  std::unique_ptr<MemoryBuffer> buf = MemoryBuffer::getMemBuffer("repl");

  GlobalVariable *g0 = createEmbeddedFB(TapirTargetID::Cuda, *m);
  g0->setName("g0");
  g0->setMetadata("some.kitsune.md", MDNode::get(ctx, {}));

  EXPECT_TRUE(m->getGlobalVariable("g0"));
  EXPECT_FALSE(m->getGlobalVariable("g1"));
  EXPECT_EQ(getNumElements(g0->getInitializer()), 0U);

  GlobalVariable *g1 = resetEmbeddedFB(*buf, *g0);
  g1->setName("g1");

  EXPECT_TRUE(g1->hasMetadata("some.kitsune.md"));
  EXPECT_FALSE(m->getGlobalVariable("g0"));
  EXPECT_TRUE(m->getGlobalVariable("g1"));
  EXPECT_TRUE(hasKitsuneFBMD(*g1, TapirTargetID::Cuda));
  EXPECT_EQ(getNumElements(g1->getInitializer()), 4U);
}

TEST(KitsuneUtils, createKernelMDGlobal) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, R"()");

  GlobalVariable *g = createKernelMDGlobal(*m, "kname");
  g->setName("g0");

  EXPECT_TRUE(m->getGlobalVariable("g0", true));
  EXPECT_TRUE(g->hasInitializer());
  EXPECT_TRUE(isa<ConstantAggregateZero>(g->getInitializer()));
  EXPECT_TRUE(hasKitsuneKernelMDMD(*g));
}

TEST(KitsuneUtils, getOrCreateGlobalString) {
  LLVMContext ctx;
  StringRef s1 = "test1";
  StringRef s2 = "test2";
  std::unique_ptr<Module> m = parseIR(ctx, R"(
    @farr = constant [2 x i32] [i32 11, i32 22]
  )");

  GlobalVariable *g1 = getOrCreateGlobalString(*m, s1);

  EXPECT_FALSE(g1->hasName());
  EXPECT_TRUE(g1->hasInitializer());
  EXPECT_TRUE(isa<ConstantDataArray>(g1->getInitializer()));
  EXPECT_EQ(cast<ConstantDataArray>(g1->getInitializer())->getAsCString(), s1);
  EXPECT_EQ(m->global_size(), 2U);

  // It doesn't matter what other parameters are passed to the function, if a
  // global string with the given initializer already exists, it will be
  // returned.
  EXPECT_EQ(getOrCreateGlobalString(*m, s1), g1);
  EXPECT_EQ(getOrCreateGlobalString(*m, s1, "newName"), g1);
  EXPECT_FALSE(getOrCreateGlobalString(*m, s1, "newName")->hasName());

  // When asking for a different string, make sure that works too.
  GlobalVariable *g2 = getOrCreateGlobalString(*m, s2, "s2");

  EXPECT_EQ(m->global_size(), 3U);
  EXPECT_EQ(g2->getName(), "s2");
}

TEST(KitsuneUtils, getTargetsGeneratingEmbBC) {
  std::array<TapirTargetID, 2> exp = {TapirTargetID::Cuda, TapirTargetID::Hip};
  EXPECT_EQ(getTargetsGeneratingEmbBC(), exp);
}

TEST(KitsuneUtils, generatesEmbBC) {
  EXPECT_FALSE(generatesEmbBC(TapirTargetID::None));
  EXPECT_FALSE(generatesEmbBC(TapirTargetID::Serial));
  EXPECT_FALSE(generatesEmbBC(TapirTargetID::OpenCilk));

  EXPECT_TRUE(generatesEmbBC(TapirTargetID::Cuda));
  EXPECT_TRUE(generatesEmbBC(TapirTargetID::Hip));
}

} // namespace

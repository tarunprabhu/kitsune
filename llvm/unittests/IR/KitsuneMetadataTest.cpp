//===- KitsuneMetadataTest.cpp - Unit tests for kitsune's metadata --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/KitsuneMetadata.h"
#include "llvm/AsmParser/Parser.h"
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

TEST(KitsuneMetadata, setAndCheckBCFB) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, R"(
@g = constant [1 x i8] zeroinitializer
@bcCuda = constant [1 x i8] zeroinitializer
@bcHip = constant [1 x i8] zeroinitializer
@fbCuda = constant [1 x i8] zeroinitializer
@fbHip = constant [1 x i8] zeroinitializer
)");

  GlobalVariable &g = *m->getGlobalVariable("g");
  GlobalVariable &bcCuda = *m->getGlobalVariable("bcCuda");
  GlobalVariable &bcHip = *m->getGlobalVariable("bcHip");
  GlobalVariable &fbCuda = *m->getGlobalVariable("fbCuda");
  GlobalVariable &fbHip = *m->getGlobalVariable("fbHip");

  EXPECT_FALSE(hasKitsuneBCMD(bcCuda));
  EXPECT_FALSE(hasKitsuneBCMD(bcHip));
  EXPECT_FALSE(hasKitsuneFBMD(fbCuda));
  EXPECT_FALSE(hasKitsuneFBMD(fbHip));

  setKitsuneBCMD(bcCuda, TapirTargetID::Cuda);
  setKitsuneBCMD(bcHip, TapirTargetID::Hip);
  setKitsuneFBMD(fbCuda, TapirTargetID::Cuda);
  setKitsuneFBMD(fbHip, TapirTargetID::Hip);

  EXPECT_TRUE(hasKitsuneBCMD(bcCuda));
  EXPECT_TRUE(hasKitsuneBCMD(bcHip));
  EXPECT_FALSE(hasKitsuneFBMD(bcCuda));
  EXPECT_FALSE(hasKitsuneFBMD(bcHip));

  EXPECT_TRUE(hasKitsuneBCMD(bcCuda, TapirTargetID::Cuda));
  EXPECT_FALSE(hasKitsuneBCMD(bcCuda, TapirTargetID::Hip));
  EXPECT_FALSE(hasKitsuneBCMD(bcHip, TapirTargetID::Cuda));
  EXPECT_TRUE(hasKitsuneBCMD(bcHip, TapirTargetID::Hip));

  EXPECT_FALSE(hasKitsuneBCMD(fbCuda));
  EXPECT_FALSE(hasKitsuneBCMD(fbHip));
  EXPECT_TRUE(hasKitsuneFBMD(fbCuda));
  EXPECT_TRUE(hasKitsuneFBMD(fbHip));

  EXPECT_TRUE(hasKitsuneFBMD(fbCuda, TapirTargetID::Cuda));
  EXPECT_FALSE(hasKitsuneFBMD(fbCuda, TapirTargetID::Hip));
  EXPECT_FALSE(hasKitsuneFBMD(fbHip, TapirTargetID::Cuda));
  EXPECT_TRUE(hasKitsuneFBMD(fbHip, TapirTargetID::Hip));

  EXPECT_EQ(getKitsuneBCMD(bcCuda), TapirTargetID::Cuda);
  EXPECT_EQ(getKitsuneBCMD(bcHip), TapirTargetID::Hip);
  EXPECT_EQ(getKitsuneBCMD(g), std::nullopt);
  EXPECT_EQ(getKitsuneBCMD(fbCuda), std::nullopt);
  EXPECT_EQ(getKitsuneBCMD(fbHip), std::nullopt);

  EXPECT_EQ(getKitsuneFBMD(fbCuda), TapirTargetID::Cuda);
  EXPECT_EQ(getKitsuneFBMD(fbHip), TapirTargetID::Hip);
  EXPECT_EQ(getKitsuneFBMD(g), std::nullopt);
  EXPECT_EQ(getKitsuneFBMD(bcCuda), std::nullopt);
  EXPECT_EQ(getKitsuneFBMD(bcHip), std::nullopt);
}

TEST(KitsuneMetadata, setAndCheckKernelMD) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, R"(
@kmd = constant {i64, i64, i64, i64} zeroinitializer
)");
  GlobalVariable &kmd = *m->getGlobalVariable("kmd");

  EXPECT_FALSE(hasKitsuneKernelMDMD(kmd));
  EXPECT_EQ(getKitsuneKernelMDMD(kmd), std::nullopt);

  setKitsuneKernelMDMD(kmd, "kfname");

  EXPECT_TRUE(hasKitsuneKernelMDMD(kmd));
  EXPECT_EQ(getKitsuneKernelMDMD(kmd), "kfname");
}

TEST(KitsuneMetadata, setAndCheckModuleMD) {
  LLVMContext ctx;
  Module m("modname", ctx);

  EXPECT_FALSE(hasKitsuneModuleMD(m));
  EXPECT_FALSE(getTapirTargetFromModuleMD(m));
  EXPECT_FALSE(getNameFromModuleMD(m));

  addKitsuneModuleMD(TapirTargetID::Serial, m);
  EXPECT_TRUE(hasKitsuneModuleMD(m));
  EXPECT_EQ(*getTapirTargetFromModuleMD(m), TapirTargetID::Serial);
  EXPECT_EQ(*getNameFromModuleMD(m), "modname");
}

} // namespace

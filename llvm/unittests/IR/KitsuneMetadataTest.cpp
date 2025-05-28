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

TEST(KitsuneMetadata, setAndCheck) {
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

  EXPECT_EQ(getKitsuneTTMD(bcCuda), TapirTargetID::Cuda);
  EXPECT_EQ(getKitsuneTTMD(bcHip), TapirTargetID::Hip);
  EXPECT_EQ(getKitsuneTTMD(fbCuda), TapirTargetID::Cuda);
  EXPECT_EQ(getKitsuneTTMD(fbHip), TapirTargetID::Hip);
  EXPECT_EQ(getKitsuneTTMD(g), std::nullopt);
}

} // namespace

//===- SCEVUtilsTest.cpp - Unit tests for Kitsune SCEV utilities ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/SCEVUtils.h"
#include "TestUtils.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

class KitSCEVUtils : public ::testing::Test {
protected:
  LLVMContext ctx;
  std::unique_ptr<Module> m = nullptr;
  Function *f = nullptr;
  std::unique_ptr<AssumptionCache> ac = nullptr;
  std::unique_ptr<DominatorTree> dt = nullptr;
  std::unique_ptr<LoopInfo> li = nullptr;
  std::unique_ptr<ScalarEvolution> se = nullptr;
  std::unique_ptr<TargetLibraryInfo> tli = nullptr;
  std::unique_ptr<TargetLibraryInfoImpl> tlii = nullptr;

protected:
  void setup(StringRef ll, StringRef fname) {
    m = parseIR(ctx, ll);
    f = m->getFunction(fname);
    ac = std::make_unique<AssumptionCache>(*f);
    dt = std::make_unique<DominatorTree>(*f);
    li = std::make_unique<LoopInfo>(*dt);
    tlii = std::make_unique<TargetLibraryInfoImpl>(m->getTargetTriple());
    tli = std::make_unique<TargetLibraryInfo>(*tlii);
    se = std::make_unique<ScalarEvolution>(*f, *tli, *ac, *dt, *li);
  }
};

static constexpr StringRef nonMatchingOperandLL = R"(
define i64 @f(i64 %a) {
  %1 = add i64 %a, 11
  ret i64 %1
}
)";

TEST_F(KitSCEVUtils, getNonMatchingOperand) {
  setup(nonMatchingOperandLL, "f");

  Argument *a = f->getArg(0);
  ReturnInst *ret = cast<ReturnInst>(f->getEntryBlock().getTerminator());
  Value *rv = ret->getOperand(0);

  const SCEV *scRet = se->getSCEV(rv);
  const SCEV *scA = se->getSCEV(a);
  const SCEV *scConst = getNonMatchingOperand(scRet, scA);

  EXPECT_FALSE(getNonMatchingOperand(scRet, scRet));
  EXPECT_TRUE(isa<SCEVConstant>(scConst));
  EXPECT_EQ(cast<SCEVConstant>(scConst)->getAPInt().getLimitedValue(), 11U);
}

} // namespace

//===- LoopUtilsTest.cpp - Unit tests for Kitsune's loop utilities --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/LoopUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace llvm;

constexpr StringRef loop1 = R"(
define void @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp.not = icmp eq i64 %n, 0
  br i1 %cmp.not, label %for.i.sync, label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
  reattach within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %exitcond.i.not = icmp eq i64 %inc.i, %n
  br i1 %exitcond.i.not, label %for.i.sync, label %for.i.header, !llvm.loop !0

for.i.sync:
  sync within %syncreg, label %for.i.exit

for.i.exit:
  ret void
}

!0 = distinct !{!0}
)";

static std::unique_ptr<Module> parseIR(LLVMContext &ctx, StringRef ir) {
  SMDiagnostic err;
  std::unique_ptr<Module> m = parseAssemblyString(ir, err, ctx);
  if (!m)
    err.print("parseIR", errs());
  return m;
}

TEST(LoopUtilsTest, loopPerfectDepthTest) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, loop1);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);

  Loop *loop = li.getLoopsInPreorder().front();
  EXPECT_EQ(getTapirLoopPerfectDepthMD(*loop), 0U);

  setTapirLoopPerfectDepthMD(*loop, 42);
  EXPECT_EQ(getTapirLoopPerfectDepthMD(*loop), 42U);

  setTapirLoopPerfectDepthMD(*loop, 97);
  EXPECT_EQ(getTapirLoopPerfectDepthMD(*loop), 97U);
}

TEST(LoopUtilsTest, loopPerfectLevelTest) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, loop1);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);

  Loop *loop = li.getLoopsInPreorder().front();
  EXPECT_EQ(getTapirLoopPerfectLevelMD(*loop), 0U);

  setTapirLoopPerfectLevelMD(*loop, 42);
  EXPECT_EQ(getTapirLoopPerfectLevelMD(*loop), 42U);

  setTapirLoopPerfectLevelMD(*loop, 97);
  EXPECT_EQ(getTapirLoopPerfectLevelMD(*loop), 97U);
}

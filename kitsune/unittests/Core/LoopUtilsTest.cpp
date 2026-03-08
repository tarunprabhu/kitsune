//===- LoopUtilsTest.cpp - Unit tests for Kitsune's loop utilities --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/LoopAttrs.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
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

struct LoopInfoContext {
  LLVMContext ctx;
  std::unique_ptr<Module> m;
  Function *f = nullptr;
  DominatorTree dt;
  LoopInfo li;

  LoopInfoContext(StringRef ir, StringRef fname)
      : m(parseIR(ctx, ir)), f(m->getFunction(fname)), dt(*f), li(dt) {}
};

static bool hasAttr(const MDNode &loopMD, StringRef attrName) {
  for (unsigned i = 1, ie = loopMD.getNumOperands(); i < ie; ++i)
    if (auto *md = dyn_cast<MDNode>(loopMD.getOperand(i)))
      if (auto *mdStr = dyn_cast<MDString>(md->getOperand(0)))
        if (mdStr->getString() == attrName)
          return true;
  return false;
}

static bool hasLoopAttr(const Loop &loop, StringRef attrName) {
  return hasAttr(*loop.getLoopID(), attrName);
}

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

TEST(KitLoopUtils, clearAttrs) {
  LoopInfoContext liCtx(loop1, "f");
  LLVMContext &ctx = liCtx.ctx;
  LoopInfo &li = liCtx.li;

  Loop *loop = li.getLoopsInPreorder().front();
  MDNode *md = MDNode::get(ctx, {MDString::get(ctx, "loop.unroll")});
  addLoweringEnabledAttr(*loop);
  addPerfectDepthAttr(*loop, 17U);
  addTargetAttr(*loop, TTID::Serial);
  loop->setLoopID(
      makePostTransformationMetadata(ctx, loop->getLoopID(), {}, {md}));

  EXPECT_TRUE(hasLoweringEnabledAttr(*loop));
  EXPECT_TRUE(hasPerfectDepthAttr(*loop));
  EXPECT_TRUE(hasTargetAttr(*loop));
  EXPECT_TRUE(hasLoopAttr(*loop, "loop.unroll"));
  EXPECT_EQ(loop->getLoopID()->getNumOperands(), 5U);

  clearTapirLoopAttrs(*loop);
  EXPECT_FALSE(hasLoweringEnabledAttr(*loop));
  EXPECT_FALSE(hasPerfectDepthAttr(*loop));
  EXPECT_FALSE(hasTargetAttr(*loop));
  EXPECT_TRUE(hasLoopAttr(*loop, "loop.unroll"));
  EXPECT_EQ(loop->getLoopID()->getNumOperands(), 2U);
}

constexpr StringRef loop3_2 = R"(
define void @f(i64 %m, i64 %n, i64 %p, i64 %q) {
entry:
  %cmp.m = icmp sgt i64 %m, 0
  %cmp.n = icmp sgt i64 %n, 0
  %cmp.p = icmp sgt i64 %p, 0
  %cmp.q = icmp sgt i64 %q, 0
  br i1 %cmp.m, label %for.i.header, label %for.i.exit

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  br i1 %cmp.n, label %for.j.header, label %for.i.latch

for.j.header:
  %j = phi i64 [ 0, %for.i.header ], [ %inc.j, %for.j.latch ]
  br i1 %cmp.p, label %for.k, label %for.k.exit

for.k:
  %k = phi i64 [ 0, %for.j.header ], [ %inc.k, %for.k ]
  %inc.k = add nuw nsw i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k, !llvm.loop !2

for.k.exit:
  br i1 %cmp.q, label %for.l, label %for.j.latch

for.l:
  %l = phi i64 [ %inc.l, %for.l ], [ 0, %for.k.exit ]
  %inc.l = add nuw nsw i64 %l, 1
  %cmp.l = icmp eq i64 %inc.l, %q
  br i1 %cmp.l, label %for.j.latch, label %for.l, !llvm.loop !3

for.j.latch:
  %inc.j = add nuw nsw i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.i.latch, label %for.j.header, !llvm.loop !1

for.i.latch:
  %inc.i = add nuw nsw i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  ret void
}

!0 = distinct !{!0}
!1 = distinct !{!1}
!2 = distinct !{!2}
!3 = distinct !{!3})";

TEST(KitLoopUtils, getAllSubLoops) {
  std::unique_ptr<LoopInfoContext> liCtx = nullptr;

  liCtx.reset(new LoopInfoContext(loop3_2, "f"));
  Loop *loopI = liCtx->li.getLoopsInPreorder()[0];
  Loop *loopJ = loopI->getSubLoops()[0];
  Loop *loopK = loopJ->getSubLoops()[0];
  Loop *loopL = loopJ->getSubLoops()[1];

  EXPECT_EQ(getAllSubLoops(*loopI).size(), 3U);
  EXPECT_EQ(getAllSubLoops(*loopJ).size(), 2U);
  EXPECT_EQ(getAllSubLoops(*loopK).size(), 0U);
  EXPECT_EQ(getAllSubLoops(*loopL).size(), 0U);

  liCtx.reset(new LoopInfoContext(loop1, "f"));
  EXPECT_EQ(getAllSubLoops(*liCtx->li.getLoopsInPreorder()[0]).size(), 0U);
}

TEST(KitLoopUtils, getBlocksNotInSubLoops) {
  std::unique_ptr<LoopInfoContext> liCtx = nullptr;
  Loop *loopI = nullptr;
  Loop *loopJ = nullptr;
  Loop *loopK = nullptr;
  Loop *loopL = nullptr;

  liCtx.reset(new LoopInfoContext(loop1, "f"));
  loopI = liCtx->li.getLoopsInPreorder()[0];
  EXPECT_EQ(getBlocksNotInSubLoops(*loopI).size(), 3U);

  liCtx.reset(new LoopInfoContext(loop3_2, "f"));
  loopI = liCtx->li.getLoopsInPreorder()[0];
  loopJ = loopI->getSubLoops()[0];
  loopK = loopJ->getSubLoops()[0];
  loopL = loopJ->getSubLoops()[1];
  EXPECT_EQ(getBlocksNotInSubLoops(*loopI).size(), 2U);
  EXPECT_EQ(getBlocksNotInSubLoops(*loopJ).size(), 3U);
  EXPECT_EQ(getBlocksNotInSubLoops(*loopK).size(), 1U);
  EXPECT_EQ(getBlocksNotInSubLoops(*loopL).size(), 1U);
}

} // namespace

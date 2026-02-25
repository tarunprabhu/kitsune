//===- TapirLoopNestAnalysisTest.cpp - Unit tests for tapir loop nests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Analysis/TapirLoopNestAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace llvm;

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
  AssumptionCache ac;
  DominatorTree dt;
  LoopInfo li;
  TaskInfo ti;

  LoopInfoContext(StringRef ir, StringRef fname)
      : m(parseIR(ctx, ir)), f(m->getFunction(fname)), ac(*f), dt(*f), li(dt) {
    ti.recalculate(*f, dt);
  }
};

// This consists of loop nests of the form:
//
// forall (i ...) {
//   forall (j ...) {
//     for (k ...) {
//       ;
//     }
//     forall (l ...) {
//       ;
//     }
//   }
// }
//
constexpr StringRef loop3MixedGPU = R"(
define void @f(i64 %m, i64 %n, i64 %p, i64 %q) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  %cmp.m = icmp sgt i64 %m, 0
  %cmp.n = icmp sgt i64 %n, 0
  %cmp.p = icmp sgt i64 %p, 0
  %cmp.q = icmp sgt i64 %q, 0
  br i1 %cmp.m, label %for.i.header, label %for.i.exit

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.j = tail call token @llvm.syncregion.start()
  br i1 %cmp.n, label %for.j.header, label %for.j.exit

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  %syncreg.l = tail call token @llvm.syncregion.start()
  br i1 %cmp.p, label %for.k.header, label %for.k.exit

for.k.header:
  %k = phi i64 [ 0, %for.j.body ], [ %inc.k, %for.k.header ]
  %inc.k = add nuw nsw i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !2

for.k.exit:
  br i1 %cmp.q, label %for.l.header, label %for.l.exit

for.l.header:
  %l = phi i64 [ 0, %for.k.exit ], [ %inc.l, %for.l.latch ]
  detach within %syncreg.l, label %for.l.body, label %for.l.latch

for.l.body:
  reattach within %syncreg.l, label %for.l.latch

for.l.latch:
  %inc.l = add nuw nsw i64 %l, 1
  %cmp.l = icmp eq i64 %inc.l, %q
  br i1 %cmp.l, label %for.l.exit, label %for.l.header, !llvm.loop !3

for.l.exit:
  sync within %syncreg.l, label %for.l.end

for.l.end:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add nuw nsw i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !1

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add nuw nsw i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

!0 = distinct !{!0, !4, !5}
!1 = distinct !{!1, !4, !5}
!2 = distinct !{!2}
!3 = distinct !{!3, !4, !5}
!4 = !{!"tapir.loop.target", i32 2}
!5 = !{!"tapir.loop.spawn.strategy", i32 4}
)";

// This consists of loop nests of the form:
//
// forall (i ...) {
//   forall (j ...) {
//     for (k ...) {
//       ;
//     }
//     forall (l ...) {
//       ;
//     }
//   }
// }
//
constexpr StringRef loop3MixedCPU = R"(
define void @f(i64 %m, i64 %n, i64 %p, i64 %q) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  %cmp.m = icmp sgt i64 %m, 0
  %cmp.n = icmp sgt i64 %n, 0
  %cmp.p = icmp sgt i64 %p, 0
  %cmp.q = icmp sgt i64 %q, 0
  br i1 %cmp.m, label %for.i.header, label %for.i.exit

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.j = tail call token @llvm.syncregion.start()
  br i1 %cmp.n, label %for.j.header, label %for.j.exit

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  %syncreg.l = tail call token @llvm.syncregion.start()
  br i1 %cmp.p, label %for.k.header, label %for.k.exit

for.k.header:
  %k = phi i64 [ 0, %for.j.body ], [ %inc.k, %for.k.header ]
  %inc.k = add nuw nsw i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !2

for.k.exit:
  br i1 %cmp.q, label %for.l.header, label %for.l.exit

for.l.header:
  %l = phi i64 [ 0, %for.k.exit ], [ %inc.l, %for.l.latch ]
  detach within %syncreg.l, label %for.l.body, label %for.l.latch

for.l.body:
  reattach within %syncreg.l, label %for.l.latch

for.l.latch:
  %inc.l = add nuw nsw i64 %l, 1
  %cmp.l = icmp eq i64 %inc.l, %q
  br i1 %cmp.l, label %for.l.exit, label %for.l.header, !llvm.loop !3

for.l.exit:
  sync within %syncreg.l, label %for.l.end

for.l.end:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add nuw nsw i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !1

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add nuw nsw i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

!0 = distinct !{!0, !4, !5}
!1 = distinct !{!1, !4, !5}
!2 = distinct !{!2}
!3 = distinct !{!3, !4, !5}
!4 = !{!"tapir.loop.target", i32 1024}
!5 = !{!"tapir.loop.spawn.strategy", i32 4}
)";

// This consists of loop nests of the form:
//
// forall (i ...) {
//   forall (j ...) {
//     for (k ...) {
//       ;
//     }
//     forall (l ...) {
//       ;
//     }
//   }
// }
//
constexpr StringRef loop3MixedMixed = R"(
define void @f(i64 %m, i64 %n, i64 %p, i64 %q) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  %cmp.m = icmp sgt i64 %m, 0
  %cmp.n = icmp sgt i64 %n, 0
  %cmp.p = icmp sgt i64 %p, 0
  %cmp.q = icmp sgt i64 %q, 0
  br i1 %cmp.m, label %for.i.header, label %for.i.exit

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.j = tail call token @llvm.syncregion.start()
  br i1 %cmp.n, label %for.j.header, label %for.j.exit

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  %syncreg.l = tail call token @llvm.syncregion.start()
  br i1 %cmp.p, label %for.k.header, label %for.k.exit

for.k.header:
  %k = phi i64 [ 0, %for.j.body ], [ %inc.k, %for.k.header ]
  %inc.k = add nuw nsw i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !2

for.k.exit:
  br i1 %cmp.q, label %for.l.header, label %for.l.exit

for.l.header:
  %l = phi i64 [ 0, %for.k.exit ], [ %inc.l, %for.l.latch ]
  detach within %syncreg.l, label %for.l.body, label %for.l.latch

for.l.body:
  reattach within %syncreg.l, label %for.l.latch

for.l.latch:
  %inc.l = add nuw nsw i64 %l, 1
  %cmp.l = icmp eq i64 %inc.l, %q
  br i1 %cmp.l, label %for.l.exit, label %for.l.header, !llvm.loop !3

for.l.exit:
  sync within %syncreg.l, label %for.l.end

for.l.end:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add nuw nsw i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !1

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add nuw nsw i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

!0 = distinct !{!0, !4, !5}
!1 = distinct !{!1, !5, !6}
!2 = distinct !{!2}
!3 = distinct !{!3, !4, !5}
!4 = !{!"tapir.loop.target", i32 2}
!5 = !{!"tapir.loop.spawn.strategy", i32 4}
!6 = !{!"tapir.loop.target", i32 4}
)";

TEST(TapirLoopNestAnalysisTest, isTapirLoop) {
  LoopInfoContext liCtx(loop3MixedGPU, "f");
  LoopInfo &li = liCtx.li;
  TaskInfo &ti = liCtx.ti;

  Loop *loopI = li.getLoopsInPreorder()[0];
  Loop *loopJ = loopI->getSubLoops()[0];
  Loop *loopK = loopJ->getSubLoops()[0];
  Loop *loopL = loopJ->getSubLoops()[1];

  EXPECT_TRUE(isTapirLoop(*loopI, ti));
  EXPECT_TRUE(isTapirLoop(*loopJ, ti));
  EXPECT_FALSE(isTapirLoop(*loopK, ti));
  EXPECT_TRUE(isTapirLoop(*loopL, ti));
}

TEST(TapirLoopNestAnalysisTest, isTopLevelTapirLoop) {
  LoopInfoContext liCtx(loop3MixedGPU, "f");
  LoopInfo &li = liCtx.li;
  TaskInfo &ti = liCtx.ti;

  Loop *loopI = li.getLoopsInPreorder()[0];
  Loop *loopJ = loopI->getSubLoops()[0];
  Loop *loopK = loopJ->getSubLoops()[0];
  Loop *loopL = loopJ->getSubLoops()[1];

  EXPECT_TRUE(isTopLevelTapirLoop(*loopI, ti));
  EXPECT_FALSE(isTopLevelTapirLoop(*loopJ, ti));
  EXPECT_FALSE(isTopLevelTapirLoop(*loopK, ti));
  EXPECT_FALSE(isTopLevelTapirLoop(*loopL, ti));
}

TEST(TapirLoopAnalysisTest, isTapirLoopForGPU) {
  {
    LoopInfoContext liCtx(loop3MixedCPU, "f");
    LoopInfo &li = liCtx.li;
    TaskInfo &ti = liCtx.ti;

    Loop *loopI = li.getLoopsInPreorder()[0];
    Loop *loopJ = loopI->getSubLoops()[0];
    Loop *loopK = loopJ->getSubLoops()[0];
    Loop *loopL = loopJ->getSubLoops()[1];

    EXPECT_FALSE(isTapirLoopForGPU(*loopI, ti));
    EXPECT_FALSE(isTapirLoopForGPU(*loopJ, ti));
    EXPECT_FALSE(isTapirLoopForGPU(*loopK, ti));
    EXPECT_FALSE(isTapirLoopForGPU(*loopL, ti));
  }

  {
    LoopInfoContext liCtx(loop3MixedGPU, "f");
    LoopInfo &li = liCtx.li;
    TaskInfo &ti = liCtx.ti;

    Loop *loopI = li.getLoopsInPreorder()[0];
    Loop *loopJ = loopI->getSubLoops()[0];
    Loop *loopK = loopJ->getSubLoops()[0];
    Loop *loopL = loopJ->getSubLoops()[1];

    EXPECT_TRUE(isTapirLoopForGPU(*loopI, ti));
    EXPECT_TRUE(isTapirLoopForGPU(*loopJ, ti));
    EXPECT_FALSE(isTapirLoopForGPU(*loopK, ti));
    EXPECT_TRUE(isTapirLoopForGPU(*loopL, ti));
  }

  {
    LoopInfoContext liCtx(loop3MixedMixed, "f");
    LoopInfo &li = liCtx.li;
    TaskInfo &ti = liCtx.ti;

    Loop *loopI = li.getLoopsInPreorder()[0];
    Loop *loopJ = loopI->getSubLoops()[0];
    Loop *loopK = loopJ->getSubLoops()[0];
    Loop *loopL = loopJ->getSubLoops()[1];

    EXPECT_FALSE(isTapirLoopForGPU(*loopI, ti));
    EXPECT_FALSE(isTapirLoopForGPU(*loopJ, ti));
    EXPECT_FALSE(isTapirLoopForGPU(*loopK, ti));
    EXPECT_TRUE(isTapirLoopForGPU(*loopL, ti));
  }
}

TEST(TapirLoopAnalysisTest, isTopLevelTapirLoopForGPU) {
  {
    LoopInfoContext liCtx(loop3MixedGPU, "f");
    LoopInfo &li = liCtx.li;
    TaskInfo &ti = liCtx.ti;

    Loop *loopI = li.getLoopsInPreorder()[0];
    Loop *loopJ = loopI->getSubLoops()[0];
    Loop *loopK = loopJ->getSubLoops()[0];
    Loop *loopL = loopJ->getSubLoops()[1];

    EXPECT_TRUE(isTopLevelTapirLoopForGPU(*loopI, ti));
    EXPECT_FALSE(isTopLevelTapirLoopForGPU(*loopJ, ti));
    EXPECT_FALSE(isTopLevelTapirLoopForGPU(*loopK, ti));
    EXPECT_FALSE(isTopLevelTapirLoopForGPU(*loopL, ti));
  }

  {
    LoopInfoContext liCtx(loop3MixedMixed, "f");
    LoopInfo &li = liCtx.li;
    TaskInfo &ti = liCtx.ti;

    Loop *loopI = li.getLoopsInPreorder()[0];
    Loop *loopJ = loopI->getSubLoops()[0];
    Loop *loopK = loopJ->getSubLoops()[0];
    Loop *loopL = loopJ->getSubLoops()[1];

    EXPECT_FALSE(isTopLevelTapirLoopForGPU(*loopI, ti));
    EXPECT_FALSE(isTopLevelTapirLoopForGPU(*loopJ, ti));
    EXPECT_FALSE(isTopLevelTapirLoopForGPU(*loopK, ti));
    EXPECT_FALSE(isTopLevelTapirLoopForGPU(*loopL, ti));
  }
}

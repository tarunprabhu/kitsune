//===- LoopUtilsTest.cpp - Unit tests for Kitsune's loop utilities --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/LoopUtils.h"
#include "Core/AttrsImpl.h"
#include "TestUtils.h"
#include "kitsune/Core/LoopAttrs.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

class KitLoopUtils : public ::testing::Test {
protected:
  LLVMContext ctx;
  std::unique_ptr<Module> m = nullptr;
  Function *f = nullptr;
  std::unique_ptr<DominatorTree> dt = nullptr;
  std::unique_ptr<LoopInfo> li = nullptr;

protected:
  void setup(StringRef ll, StringRef fname) {
    m = parseIR(ctx, ll);
    f = m->getFunction(fname);
    dt = std::make_unique<DominatorTree>(*f);
    li = std::make_unique<LoopInfo>(*dt);
  }
};

static bool hasLoopAttr(const Loop &loop, StringRef attrName) {
  return detail::getRawAttr(attrName, loop.getLoopID());
}

static constexpr StringRef loop1 = R"(
define void @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.i.header

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

TEST_F(KitLoopUtils, clearAttrs) {
  setup(loop1, "f");

  Loop *loop = li->getLoopsInPreorder().front();
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

static constexpr StringRef loop3_2 = R"(
define void @f(i64 %m, i64 %n, i64 %p, i64 %q) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.header ], [ %inc.j, %for.j.latch ]
  br label %for.k

for.k:
  %k = phi i64 [ 0, %for.j.header ], [ %inc.k, %for.k ]
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k, !llvm.loop !2

for.k.exit:
  br label %for.l

for.l:
  %l = phi i64 [ %inc.l, %for.l ], [ 0, %for.k.exit ]
  %inc.l = add i64 %l, 1
  %cmp.l = icmp eq i64 %inc.l, %q
  br i1 %cmp.l, label %for.j.latch, label %for.l, !llvm.loop !3

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.i.latch, label %for.j.header, !llvm.loop !1

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  ret void
}

!0 = distinct !{!0}
!1 = distinct !{!1}
!2 = distinct !{!2}
!3 = distinct !{!3})";

TEST_F(KitLoopUtils, getAllSubLoops) {
  Loop *loopI = nullptr;
  Loop *loopJ = nullptr;
  Loop *loopK = nullptr;
  Loop *loopL = nullptr;

  setup(loop3_2, "f");
  loopI = li->getLoopsInPreorder()[0];
  loopJ = loopI->getSubLoops()[0];
  loopK = loopJ->getSubLoops()[0];
  loopL = loopJ->getSubLoops()[1];

  EXPECT_EQ(getAllSubLoops(*loopI).size(), 3U);
  EXPECT_EQ(getAllSubLoops(*loopJ).size(), 2U);
  EXPECT_EQ(getAllSubLoops(*loopK).size(), 0U);
  EXPECT_EQ(getAllSubLoops(*loopL).size(), 0U);

  setup(loop1, "f");
  EXPECT_EQ(getAllSubLoops(**li->begin()).size(), 0U);
}

TEST_F(KitLoopUtils, getBlocksNotInSubLoops) {
  Loop *loopI = nullptr;
  Loop *loopJ = nullptr;
  Loop *loopK = nullptr;
  Loop *loopL = nullptr;

  setup(loop1, "f");
  loopI = li->getLoopsInPreorder()[0];

  EXPECT_EQ(getBlocksNotInSubLoops(*loopI).size(), 3U);

  setup(loop3_2, "f");
  loopI = li->getLoopsInPreorder()[0];
  loopJ = loopI->getSubLoops()[0];
  loopK = loopJ->getSubLoops()[0];
  loopL = loopJ->getSubLoops()[1];

  EXPECT_EQ(getBlocksNotInSubLoops(*loopI).size(), 2U);
  EXPECT_EQ(getBlocksNotInSubLoops(*loopJ).size(), 3U);
  EXPECT_EQ(getBlocksNotInSubLoops(*loopK).size(), 1U);
  EXPECT_EQ(getBlocksNotInSubLoops(*loopL).size(), 1U);
}

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
static constexpr StringRef loop3MixedGPU = R"(
define void @f(i64 %m, i64 %n, i64 %p, i64 %q) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.j = tail call token @llvm.syncregion.start()
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  %syncreg.l = tail call token @llvm.syncregion.start()
  br label %for.k.header

for.k.header:
  %k = phi i64 [ 0, %for.j.body ], [ %inc.k, %for.k.header ]
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !2

for.k.exit:
  br label %for.l.header

for.l.header:
  %l = phi i64 [ 0, %for.k.exit ], [ %inc.l, %for.l.latch ]
  detach within %syncreg.l, label %for.l.body, label %for.l.latch

for.l.body:
  reattach within %syncreg.l, label %for.l.latch

for.l.latch:
  %inc.l = add i64 %l, 1
  %cmp.l = icmp eq i64 %inc.l, %q
  br i1 %cmp.l, label %for.l.exit, label %for.l.header, !llvm.loop !3

for.l.exit:
  sync within %syncreg.l, label %for.l.end

for.l.end:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !1

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

!0 = distinct !{!0, !4}
!1 = distinct !{!1, !4}
!2 = distinct !{!2}
!3 = distinct !{!3, !4}
!4 = !{!"tapir.loop.target", i32 2}
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
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.j = tail call token @llvm.syncregion.start()
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  %syncreg.l = tail call token @llvm.syncregion.start()
  br label %for.k.header

for.k.header:
  %k = phi i64 [ 0, %for.j.body ], [ %inc.k, %for.k.header ]
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !2

for.k.exit:
  br label %for.l.header

for.l.header:
  %l = phi i64 [ 0, %for.k.exit ], [ %inc.l, %for.l.latch ]
  detach within %syncreg.l, label %for.l.body, label %for.l.latch

for.l.body:
  reattach within %syncreg.l, label %for.l.latch

for.l.latch:
  %inc.l = add i64 %l, 1
  %cmp.l = icmp eq i64 %inc.l, %q
  br i1 %cmp.l, label %for.l.exit, label %for.l.header, !llvm.loop !3

for.l.exit:
  sync within %syncreg.l, label %for.l.end

for.l.end:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !1

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

!0 = distinct !{!0, !4}
!1 = distinct !{!1, !4}
!2 = distinct !{!2}
!3 = distinct !{!3, !4}
!4 = !{!"tapir.loop.target", i32 1024}
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
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.j = tail call token @llvm.syncregion.start()
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  %syncreg.l = tail call token @llvm.syncregion.start()
  br label %for.k.header

for.k.header:
  %k = phi i64 [ 0, %for.j.body ], [ %inc.k, %for.k.header ]
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !2

for.k.exit:
  br label %for.l.header

for.l.header:
  %l = phi i64 [ 0, %for.k.exit ], [ %inc.l, %for.l.latch ]
  detach within %syncreg.l, label %for.l.body, label %for.l.latch

for.l.body:
  reattach within %syncreg.l, label %for.l.latch

for.l.latch:
  %inc.l = add i64 %l, 1
  %cmp.l = icmp eq i64 %inc.l, %q
  br i1 %cmp.l, label %for.l.exit, label %for.l.header, !llvm.loop !3

for.l.exit:
  sync within %syncreg.l, label %for.l.end

for.l.end:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !1

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

!0 = distinct !{!0, !4}
!1 = distinct !{!1, !5}
!2 = distinct !{!2}
!3 = distinct !{!3, !4}
!4 = !{!"tapir.loop.target", i32 2}
!5 = !{!"tapir.loop.target", i32 4}
)";

// This consists of two loops of the form
//
// forall (i ...) {
//   ;
// }
// forall (i2 ...) {
//   ;
// }
//
static constexpr StringRef loop2 = R"(
define void @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
  reattach within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  sync within %syncreg, label %for.i2.header

for.i2.header:
  %i2 = phi i64 [ 0, %for.i.exit ], [ %inc.i2, %for.i2.latch ]
  detach within %syncreg, label %for.i2.body, label %for.i2.latch

for.i2.body:
  reattach within %syncreg, label %for.i2.latch

for.i2.latch:
  %inc.i2 = add i64 %i2, 1
  %cmp.i2 = icmp eq i64 %inc.i2, %n
  br i1 %cmp.i2, label %for.i2.exit, label %for.i2.header, !llvm.loop !2

for.i2.exit:
  sync within %syncreg, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 1024}
!2 = distinct !{!2, !1}
)";

TEST_F(KitLoopUtils, isTapirLoop) {
  setup(loop3MixedGPU, "f");
  Loop *loopI = li->getLoopsInPreorder()[0];
  Loop *loopJ = loopI->getSubLoops()[0];
  Loop *loopK = loopJ->getSubLoops()[0];
  Loop *loopL = loopJ->getSubLoops()[1];

  EXPECT_TRUE(isTapirLoop(*loopI));
  EXPECT_TRUE(isTapirLoop(*loopJ));
  EXPECT_FALSE(isTapirLoop(*loopK));
  EXPECT_TRUE(isTapirLoop(*loopL));
}

TEST_F(KitLoopUtils, isTopLevelTapirLoop) {
  setup(loop3MixedGPU, "f");
  Loop *loopI = li->getLoopsInPreorder()[0];
  Loop *loopJ = loopI->getSubLoops()[0];
  Loop *loopK = loopJ->getSubLoops()[0];
  Loop *loopL = loopJ->getSubLoops()[1];

  EXPECT_TRUE(isTopLevelTapirLoop(*loopI));
  EXPECT_FALSE(isTopLevelTapirLoop(*loopJ));
  EXPECT_FALSE(isTopLevelTapirLoop(*loopK));
  EXPECT_FALSE(isTopLevelTapirLoop(*loopL));
}

TEST_F(KitLoopUtils, isTapirLoopForGPU) {
  Loop *loopI = nullptr;
  Loop *loopJ = nullptr;
  Loop *loopK = nullptr;
  Loop *loopL = nullptr;

  setup(loop3MixedCPU, "f");
  loopI = li->getLoopsInPreorder()[0];
  loopJ = loopI->getSubLoops()[0];
  loopK = loopJ->getSubLoops()[0];
  loopL = loopJ->getSubLoops()[1];

  EXPECT_FALSE(isTapirLoopForGPU(*loopI));
  EXPECT_FALSE(isTapirLoopForGPU(*loopJ));
  EXPECT_FALSE(isTapirLoopForGPU(*loopK));
  EXPECT_FALSE(isTapirLoopForGPU(*loopL));

  setup(loop3MixedGPU, "f");
  loopI = li->getLoopsInPreorder()[0];
  loopJ = loopI->getSubLoops()[0];
  loopK = loopJ->getSubLoops()[0];
  loopL = loopJ->getSubLoops()[1];

  EXPECT_TRUE(isTapirLoopForGPU(*loopI));
  EXPECT_TRUE(isTapirLoopForGPU(*loopJ));
  EXPECT_FALSE(isTapirLoopForGPU(*loopK));
  EXPECT_TRUE(isTapirLoopForGPU(*loopL));

  setup(loop3MixedMixed, "f");
  loopI = li->getLoopsInPreorder()[0];
  loopJ = loopI->getSubLoops()[0];
  loopK = loopJ->getSubLoops()[0];
  loopL = loopJ->getSubLoops()[1];

  EXPECT_FALSE(isTapirLoopForGPU(*loopI));
  EXPECT_FALSE(isTapirLoopForGPU(*loopJ));
  EXPECT_FALSE(isTapirLoopForGPU(*loopK));
  EXPECT_TRUE(isTapirLoopForGPU(*loopL));
}

TEST_F(KitLoopUtils, isTopLevelTapirLoopForGPU) {
  Loop *loopI = nullptr;
  Loop *loopJ = nullptr;
  Loop *loopK = nullptr;
  Loop *loopL = nullptr;

  setup(loop3MixedGPU, "f");

  loopI = li->getLoopsInPreorder()[0];
  loopJ = loopI->getSubLoops()[0];
  loopK = loopJ->getSubLoops()[0];
  loopL = loopJ->getSubLoops()[1];

  EXPECT_TRUE(isTopLevelTapirLoopForGPU(*loopI));
  EXPECT_FALSE(isTopLevelTapirLoopForGPU(*loopJ));
  EXPECT_FALSE(isTopLevelTapirLoopForGPU(*loopK));
  EXPECT_FALSE(isTopLevelTapirLoopForGPU(*loopL));

  setup(loop3MixedMixed, "f");

  loopI = li->getLoopsInPreorder()[0];
  loopJ = loopI->getSubLoops()[0];
  loopK = loopJ->getSubLoops()[0];
  loopL = loopJ->getSubLoops()[1];

  EXPECT_FALSE(isTopLevelTapirLoopForGPU(*loopI));
  EXPECT_FALSE(isTopLevelTapirLoopForGPU(*loopJ));
  EXPECT_FALSE(isTopLevelTapirLoopForGPU(*loopK));
  EXPECT_FALSE(isTopLevelTapirLoopForGPU(*loopL));
}

TEST_F(KitLoopUtils, getTopLevelTapirLoops) {
  setup(loop2, "f");
  EXPECT_EQ(getTopLevelTapirLoops(*li).size(), 2U);

  setup(loop3MixedCPU, "f");
  EXPECT_EQ(getTopLevelTapirLoops(*li).size(), 1U);
}

TEST_F(KitLoopUtils, getTapirLoops) {
  setup(loop2, "f");
  EXPECT_EQ(getTapirLoops(*li).size(), 2U);

  setup(loop3MixedCPU, "f");
  EXPECT_EQ(getTapirLoops(*li).size(), 3U);
}

static constexpr StringRef simpleTapirLoop = R"(
define void @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
  reattach within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  sync within %syncreg, label %for.i.end

for.i.end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 1}
)";

TEST_F(KitLoopUtils, addMandatoryLLVMLoopAttrs) {
  setup(simpleTapirLoop, "f");
  Loop *loop = *li->begin();

  // The loop does not contain any attributes. The loop metadata should contain
  // two single operands - a reference to itself, and the tapir loop target.
  EXPECT_EQ(loop->getLoopID()->getNumOperands(), 2U);
  EXPECT_TRUE(hasTargetAttr(*loop));

  addMandatoryLLVMLoopAttrs(*loop);

  // Check for each operand that is expected to be present after the mandatory
  // attributes have been added.
  EXPECT_EQ(loop->getLoopID()->getNumOperands(), 4U);
  EXPECT_TRUE(hasTargetAttr(*loop));
  EXPECT_TRUE(hasLoopAttr(*loop, "llvm.loop.unroll.disable"));
  EXPECT_TRUE(hasLoopAttr(*loop, "llvm.loop.vectorize.enable"));

  // This should not change the number of loop operands since the mandatory
  // attributes have already been added.
  addMandatoryLLVMLoopAttrs(*loop);
  EXPECT_EQ(loop->getLoopID()->getNumOperands(), 4U);
}

TEST_F(KitLoopUtils, clearMandatoryLLVMLoopAttrs) {
  setup(simpleTapirLoop, "f");
  Loop *loop = *li->begin();

  // The loop does not contain any attributes. The loop metadata should contain
  // two single operands - a reference to itself, and the tapir loop target.
  EXPECT_EQ(loop->getLoopID()->getNumOperands(), 2U);
  EXPECT_TRUE(hasTargetAttr(*loop));

  // Sanity check that the number of attributes is as expected after they have
  // been added.
  addMandatoryLLVMLoopAttrs(*loop);
  EXPECT_EQ(loop->getLoopID()->getNumOperands(), 4U);

  // Now remove them and we should get what we originally had.
  clearMandatoryLLVMLoopAttrs(*loop);
  EXPECT_EQ(loop->getLoopID()->getNumOperands(), 2U);
  EXPECT_TRUE(hasTargetAttr(*loop));

  // This should not change the number of loop operands since the mandatory
  // attributes have already been removed.
  clearMandatoryLLVMLoopAttrs(*loop);
  EXPECT_EQ(loop->getLoopID()->getNumOperands(), 2U);
}

static constexpr StringRef multipleIVs = R"(
define void @f(i64 %n) {
entry:
  br label %loop2

loop2:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %loop2 ]
  %j = phi i32 [ 1, %entry ], [ %inc.j, %loop2 ]
  %inc.i = add i64 %i, 1
  %inc.j = add i32 %j, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %loop1, label %loop2, !llvm.loop !2

loop1:
  %k = phi i64 [ 0, %entry ], [ %inc.k, %loop1 ]
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %n
  br i1 %cmp.k, label %loop0, label %loop1, !llvm.loop !1

loop0:
  br label %loop0, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0}
!1 = distinct !{!1}
!2 = distinct !{!2}
)";

TEST_F(KitLoopUtils, getNumIndVars) {
  setup(multipleIVs, "f");

  for (BasicBlock &bb : *f) {
    StringRef name = bb.getName();
    if (name == "loop0")
      EXPECT_EQ(getNumIndVars(*li->getLoopFor(&bb)), 0U);
    else if (name == "loop1")
      EXPECT_EQ(getNumIndVars(*li->getLoopFor(&bb)), 1U);
    else if (name == "loop2")
      EXPECT_EQ(getNumIndVars(*li->getLoopFor(&bb)), 2U);
  }
}

static constexpr StringRef loops = R"(
define void @f(i64 %n) {
entry:
  br label %header.o

header.o:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch.o ]
  br label %loop.i

loop.i:
  %j = phi i64 [ 0, %header.o ], [ %inc.j, %loop.i ]
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %latch.o, label %loop.i, !llvm.loop !1

latch.o:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %exit, label %header.o, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0}
!1 = distinct !{!1}
)";

TEST_F(KitLoopUtils, isInLoop) {
  setup(loops, "f");

  Loop *outerLoop, *innerLoop;
  for (BasicBlock &bb : *f)
    if (bb.getName() == "header.o")
      outerLoop = li->getLoopFor(&bb);
    else if (bb.getName() == "loop.i")
      innerLoop = li->getLoopFor(&bb);

  // latchI is the latch of the inner loop and must be in both loops, unless
  // strict is true in which case, it will only be in the inner loop.
  Instruction *latchI = innerLoop->getLoopLatch()->getTerminator();

  // By default, strict should be `false`.
  EXPECT_TRUE(isInLoop(*latchI, *outerLoop, *li));
  EXPECT_TRUE(isInLoop(*latchI, *innerLoop, *li));

  EXPECT_TRUE(isInLoop(*latchI, *outerLoop, *li, /*strict=*/false));
  EXPECT_FALSE(isInLoop(*latchI, *outerLoop, *li, /*strict=*/true));
  EXPECT_TRUE(isInLoop(*latchI, *innerLoop, *li, /*strict=*/false));
  EXPECT_TRUE(isInLoop(*latchI, *innerLoop, *li, /*strict=*/true));

  // latchO is the latch of the outer loop and must be in the outer loop, but
  // not the inner.
  Instruction *latchO = outerLoop->getLoopLatch()->getTerminator();
  EXPECT_TRUE(isInLoop(*latchO, *outerLoop, *li));
  EXPECT_TRUE(isInLoop(*latchO, *outerLoop, *li, /*strict=*/true));
  EXPECT_FALSE(isInLoop(*latchO, *innerLoop, *li));
  EXPECT_FALSE(isInLoop(*latchO, *innerLoop, *li, /*strict=*/true));

  // br has a parent, but it is not in any loop.
  Instruction *br = f->getEntryBlock().getTerminator();
  EXPECT_FALSE(isInLoop(*br, *outerLoop, *li, /*strict=*/false));
  EXPECT_FALSE(isInLoop(*br, *outerLoop, *li, /*strict=*/true));
  EXPECT_FALSE(isInLoop(*br, *innerLoop, *li, /*strict=*/false));
  EXPECT_FALSE(isInLoop(*br, *innerLoop, *li, /*strict=*/true));

  // ret does not have a parent.
  ReturnInst *ret = ReturnInst::Create(ctx);
  EXPECT_FALSE(isInLoop(*ret, *outerLoop, *li, /*strict=*/false));
  EXPECT_FALSE(isInLoop(*ret, *outerLoop, *li, /*strict=*/true));
  EXPECT_FALSE(isInLoop(*ret, *innerLoop, *li, /*strict=*/false));
  EXPECT_FALSE(isInLoop(*ret, *innerLoop, *li, /*strict=*/true));
}

static constexpr StringRef usedOutside = R"(
define i64 @f(i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 19, %entry ], [ %inc.i, %loop ]
  %j = phi i64 [ 0, %entry ], [ %inc.j, %loop ]
  %inc.i = add i64 %i, 10
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %exit, label %loop, !llvm.loop !0

exit:
  %tmp = phi i64 [ %j, %loop ]
  ret i64 %tmp
}

!0 = distinct !{!0}
)";

TEST_F(KitLoopUtils, isUsedOutsideLoop) {
  setup(usedOutside, "f");
  Loop *loop = *li->begin();
  BasicBlock *header = loop->getHeader();
  PHINode *iv = loop->getCanonicalInductionVariable();

  for (PHINode &phi : header->phis()) {
    // The loop has two induction variables. The canonical one is used outside
    // the loop, the other is not.
    EXPECT_EQ(isUsedOutsideLoop(phi, *loop, *li), &phi == iv);
  }
}

TEST_F(KitLoopUtils, getTapirLoopDetachedBlock) {
  setup(simpleTapirLoop, "f");
  Loop *loop = *li->begin();

  EXPECT_EQ(getTapirLoopDetachedBlock(*loop)->getName(), "for.i.body");
}

TEST_F(KitLoopUtils, getTapirLoopSyncRegion) {
  setup(simpleTapirLoop, "f");
  Loop *loop = *li->begin();

  EXPECT_EQ(getTapirLoopSyncRegion(*loop)->getName(), "syncreg");
}

static constexpr StringRef deadends = R"(
define void @f(i64 %m, i64 %n) {
entry:
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch ]
  br label %assert

assert:
  %chk = icmp eq i64 %i, %m
  br i1 %chk, label %whoops, label %latch

latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %exit, label %header, !llvm.loop !0

whoops:
  br label %unreachable

unreachable:
  unreachable

exit:
  ret void
}

!0 = distinct !{!0}
)";

TEST_F(KitLoopUtils, getUniqueNonDeadEndExitBlocks) {
  setup(deadends, "f");
  Loop *loop = *li->begin();

  BasicBlock *exitBlock = getUniqueNonDeadEndExitBlock(*loop);
  EXPECT_FALSE(loop->getExitBlock());
  EXPECT_TRUE(exitBlock);
  EXPECT_EQ(exitBlock->getName(), StringRef("exit"));
}

TEST_F(KitLoopUtils, getUniqueNonDeadEndExitingBlocks) {
  setup(deadends, "f");
  Loop *loop = *li->begin();

  BasicBlock *exitBlock = getUniqueNonDeadEndExitingBlock(*loop);
  EXPECT_FALSE(loop->getExitingBlock());
  EXPECT_TRUE(exitBlock);
  EXPECT_EQ(exitBlock->getName(), StringRef("latch"));
}

TEST_F(KitLoopUtils, getExitBlockFromLatch) {
  setup(deadends, "f");
  Loop *loop = *li->begin();

  BasicBlock *exit = getExitBlockFromLatch(*loop);
  EXPECT_TRUE(exit);
  EXPECT_EQ(exit->getName(), StringRef("exit"));
}

static constexpr StringRef normalizedNotTapir = R"(
define void @f(i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0}
)";

static constexpr StringRef normalizedNotSimplified = R"(
define void @f(i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  %skip = icmp sge i64 %n, 0
  br i1 %skip, label %header, label %sync

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %sync, label %header, !llvm.loop !0

sync:
  sync within %sr, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 1}
)";

static constexpr StringRef normalizedNotLCSSA = R"(
define i64 @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %entry ], [ %inc.j, %for.j.latch ]
  detach within %syncreg, label %for.j.body, label %for.j.latch

for.j.body:
  reattach within %syncreg, label %for.j.latch

for.j.latch:
  %s = sub i64 %j, 11
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !0

for.j.exit:
  sync within %syncreg, label %for.j.end

for.j.end:
  ret i64 %s
}

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 1}
)";

static constexpr StringRef normalizedNoCanon = R"(
define void @f(i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 1, %entry ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %i, %n
  br i1 %cmp, label %sync, label %header, !llvm.loop !0

sync:
  sync within %sr, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 1}
)";

static constexpr StringRef normalizedBadCmp = R"(
define void @f(i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %i, %n
  br i1 %cmp, label %sync, label %header, !llvm.loop !0

sync:
  sync within %sr, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 1}
)";

static constexpr StringRef normalizedNormalized = R"(
define void @f(i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %sync, label %header, !llvm.loop !0

sync:
  sync within %sr, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 1}
)";

TEST_F(KitLoopUtils, isTapirLoopNormalized) {
  {
    setup(normalizedNotTapir, "f");
    EXPECT_FALSE(isTapirLoopNormalized(**li->begin(), *dt));
  }
  {
    setup(normalizedNotSimplified, "f");
    EXPECT_FALSE(isTapirLoopNormalized(**li->begin(), *dt));
  }
  {
    setup(normalizedNotLCSSA, "f");
    EXPECT_FALSE(isTapirLoopNormalized(**li->begin(), *dt));
  }
  {
    setup(normalizedNoCanon, "f");
    EXPECT_FALSE(isTapirLoopNormalized(**li->begin(), *dt));
  }
  {
    setup(normalizedBadCmp, "f");
    EXPECT_FALSE(isTapirLoopNormalized(**li->begin(), *dt));
  }
  {
    setup(normalizedNormalized, "f");
    EXPECT_TRUE(isTapirLoopNormalized(**li->begin(), *dt));
  }
}

static constexpr StringRef nest1 = R"(
define void @f(i64 %n) {
entry:
  %sr1 = call token @llvm.syncregion.start();
  br label %header1

header1:
  %i1 = phi i64 [ 0, %entry ], [ %inc1, %latch1 ]
  detach within %sr1, label %body1, label %latch1

body1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n
  br i1 %cmp1, label %sync1, label %header1, !llvm.loop !3

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1}
!1 = !{!"tapir.loop.perfect.depth", i32 1}
!2 = !{!"tapir.loop.perfect.level", i32 1}
!3 = distinct !{!3, !0, !1, !2}
)";

static constexpr StringRef nest2 = R"(
define void @f(i64 %n) {
entry:
  %sr1 = call token @llvm.syncregion.start()
  br label %header1

header1:
  %i1 = phi i64 [ 0, %entry ], [ %inc1, %latch1 ]
  detach within %sr1, label %body1, label %latch1

body1:
  %sr2 = call token @llvm.syncregion.start()
  br label %header2

header2:
  %i2 = phi i64 [ 0, %body1 ], [ %inc2, %latch2 ]
  detach within %sr2, label %body2, label %latch2

body2:
  reattach within %sr2, label %latch2

latch2:
  %inc2 = add i64 %i2, 1
  %cmp2 = icmp eq i64 %inc2, %n
  br i1 %cmp2, label %sync2, label %header2, !llvm.loop !4

sync2:
  sync within %sr2, label %reattach1

reattach1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n
  br i1 %cmp1, label %sync1, label %header1, !llvm.loop !5

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1}
!1 = !{!"tapir.loop.perfect.depth", i32 2}
!2 = !{!"tapir.loop.perfect.level", i32 2}
!3 = !{!"tapir.loop.perfect.level", i32 1}
!4 = distinct !{!4, !0, !2}
!5 = distinct !{!5, !0, !1, !3}
)";

static constexpr StringRef nest3 = R"(
define void @f(i64 %n) {
entry:
  %sr1 = call token @llvm.syncregion.start()
  br label %header1

header1:
  %i1 = phi i64 [ 0, %entry ], [ %inc1, %latch1 ]
  detach within %sr1, label %body1, label %latch1

body1:
  %sr2 = call token @llvm.syncregion.start()
  br label %header2

header2:
  %i2 = phi i64 [ 0, %body1 ], [ %inc2, %latch2 ]
  detach within %sr2, label %body2, label %latch2

body2:
  %sr3 = call token @llvm.syncregion.start()
  br label %header3

header3:
  %i3 = phi i64 [ 0, %body2 ], [ %inc3, %latch3 ]
  detach within %sr3, label %body3, label %latch3

body3:
  reattach within %sr3, label %latch3

latch3:
  %inc3 = add i64 %i3, 1
  %cmp3 = icmp eq i64 %inc3, %n
  br i1 %cmp3, label %sync3, label %header3, !llvm.loop !5

sync3:
  sync within %sr3, label %reattach2

reattach2:
  reattach within %sr2, label %latch2

latch2:
  %inc2 = add i64 %i2, 1
  %cmp2 = icmp eq i64 %inc2, %n
  br i1 %cmp2, label %sync2, label %header2, !llvm.loop !6

sync2:
  sync within %sr2, label %reattach1

reattach1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n
  br i1 %cmp1, label %sync1, label %header1, !llvm.loop !7

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1}
!1 = !{!"tapir.loop.perfect.depth", i32 3}
!2 = !{!"tapir.loop.perfect.level", i32 3}
!3 = !{!"tapir.loop.perfect.level", i32 2}
!4 = !{!"tapir.loop.perfect.level", i32 1}
!5 = distinct !{!5, !0, !2}
!6 = distinct !{!6, !0, !3}
!7 = distinct !{!7, !0, !1, !4}
)";

static constexpr StringRef nest4 = R"(
define void @f(i64 %n) {
entry:
  %sr1 = call token @llvm.syncregion.start()
  br label %header1

header1:
  %i1 = phi i64 [ 0, %entry ], [ %inc1, %latch1 ]
  detach within %sr1, label %body1, label %latch1

body1:
  %sr2 = call token @llvm.syncregion.start()
  br label %header2

header2:
  %i2 = phi i64 [ 0, %body1 ], [ %inc2, %latch2 ]
  detach within %sr2, label %body2, label %latch2

body2:
  %sr3 = call token @llvm.syncregion.start()
  br label %header3

header3:
  %i3 = phi i64 [ 0, %body2 ], [ %inc3, %latch3 ]
  detach within %sr3, label %body3, label %latch3

body3:
  %sr4 = call token @llvm.syncregion.start()
  br label %header4

header4:
  %i4 = phi i64 [ 0, %body3 ], [ %inc4, %latch4 ]
  detach within %sr4, label %body4, label %latch4

body4:
  reattach within %sr4, label %latch4

latch4:
  %inc4 = add i64 %i4, 1
  %cmp4 = icmp eq i64 %inc4, %n
  br i1 %cmp4, label %sync4, label %header4, !llvm.loop !6

sync4:
  sync within %sr4, label %reattach3

reattach3:
  reattach within %sr3, label %latch3

latch3:
  %inc3 = add i64 %i3, 1
  %cmp3 = icmp eq i64 %inc3, %n
  br i1 %cmp3, label %sync3, label %header3, !llvm.loop !7

sync3:
  sync within %sr3, label %reattach2

reattach2:
  reattach within %sr2, label %latch2

latch2:
  %inc2 = add i64 %i2, 1
  %cmp2 = icmp eq i64 %inc2, %n
  br i1 %cmp2, label %sync2, label %header2, !llvm.loop !8

sync2:
  sync within %sr2, label %reattach1

reattach1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n
  br i1 %cmp1, label %sync1, label %header1, !llvm.loop !9

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1}
!1 = !{!"tapir.loop.perfect.depth", i32 4}
!2 = !{!"tapir.loop.perfect.level", i32 4}
!3 = !{!"tapir.loop.perfect.level", i32 3}
!4 = !{!"tapir.loop.perfect.level", i32 2}
!5 = !{!"tapir.loop.perfect.level", i32 1}
!6 = distinct !{!6, !0, !2}
!7 = distinct !{!7, !0, !3}
!8 = distinct !{!8, !0, !4}
!9 = distinct !{!9, !0, !1, !5}
)";

TEST_F(KitLoopUtils, getTapirLoopsInNest) {
  {
    setup(nest1, "f");
    Loop &root = **li->begin();
    SmallVector<Loop *, 4> loops = getTapirLoopsInNest(root);
    EXPECT_EQ(loops.size(), 1U);
    EXPECT_EQ(getName(*loops[0]), "header1");
  }
  {
    setup(nest2, "f");
    Loop &root = **li->begin();
    SmallVector<Loop *, 4> loops = getTapirLoopsInNest(root);
    EXPECT_EQ(loops.size(), 2U);
    EXPECT_EQ(getName(*loops[0]), "header1");
    EXPECT_EQ(getName(*loops[1]), "header2");
  }
  {
    setup(nest3, "f");
    Loop &root = **li->begin();
    SmallVector<Loop *, 4> loops = getTapirLoopsInNest(root);
    EXPECT_EQ(loops.size(), 3U);
    EXPECT_EQ(getName(*loops[0]), "header1");
    EXPECT_EQ(getName(*loops[1]), "header2");
    EXPECT_EQ(getName(*loops[2]), "header3");
  }
  {
    setup(nest4, "f");
    Loop &root = **li->begin();
    SmallVector<Loop *, 4> loops = getTapirLoopsInNest(root);
    EXPECT_EQ(loops.size(), 4U);
    EXPECT_EQ(getName(*loops[0]), "header1");
    EXPECT_EQ(getName(*loops[1]), "header2");
    EXPECT_EQ(getName(*loops[2]), "header3");
    EXPECT_EQ(getName(*loops[3]), "header4");
  }
}

} // namespace

//===- TapirTargetAnalysisTest.cpp - TapirTargetAnalysis tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Analysis/TapirTargetAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Frontend/Driver/KitsuneOptions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace llvm;

constexpr StringRef moduleNoHints = R"m(
target triple = "x86_64-unknown-linux-gnu"

define void @f(ptr nocapture noundef writeonly %c, i64 noundef %n) #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp5 = icmp sgt i64 %n, 0
  br i1 %cmp5, label %detach.preheader, label %sync

detach.preheader:
  br label %detach

detach:
  %iv = phi i64 [ 0, %detach.preheader ], [ %iv.next, %inc ]
  %iv.next = add nuw nsw i64 %iv, 1
  detach within %syncreg, label %body, label %inc

body:
  %arrayidx = getelementptr inbounds i64, ptr %c, i64 %iv
  store i64 %n, ptr %arrayidx, align 4
  reattach within %syncreg, label %inc

inc:
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %sync, label %detach, !llvm.loop !0

sync:
  sync within %syncreg, label %end

end:
  br label %entry2

entry2:
  %syncreg2 = tail call token @llvm.syncregion.start()
  %cmp6 = icmp sgt i64 %n, 0
  br i1 %cmp6, label %detach.preheader2, label %sync2

detach.preheader2:
  br label %detach2

detach2:
  %iv2 = phi i64 [ 0, %detach.preheader2 ], [ %iv.next2, %inc2 ]
  %iv.next2 = add nuw nsw i64 %iv2, 1
  detach within %syncreg2, label %body2, label %inc2

body2:
  %arrayidx2 = getelementptr inbounds i64, ptr %c, i64 %iv2
  store i64 %n, ptr %arrayidx2, align 4
  reattach within %syncreg2, label %inc2

inc2:
  %exitcond.not2 = icmp eq i64 %iv.next2, %n
  br i1 %exitcond.not2, label %sync2, label %detach2, !llvm.loop !0

sync2:
  sync within %syncreg2, label %end2

end2:
  ret void
}

declare token @llvm.syncregion.start() #1

attributes #0 = { nounwind memory(argmem: write) uwtable }
attributes #1 = { mustprogress nounwind willreturn memory(argmem: readwrite) }

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}
)m";

constexpr StringRef moduleWithHintsMixed = R"m(
target triple = "x86_64-unknown-linux-gnu"

define void @f(ptr nocapture noundef writeonly %c, i64 noundef %n) #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp5 = icmp sgt i64 %n, 0
  br i1 %cmp5, label %detach.preheader, label %sync

detach.preheader:
  br label %detach

detach:
  %iv = phi i64 [ 0, %detach.preheader ], [ %iv.next, %inc ]
  %iv.next = add nuw nsw i64 %iv, 1
  detach within %syncreg, label %body, label %inc

body:
  %arrayidx = getelementptr inbounds i64, ptr %c, i64 %iv
  store i64 %n, ptr %arrayidx, align 4
  reattach within %syncreg, label %inc

inc:
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %sync, label %detach, !llvm.loop !0

sync:
  sync within %syncreg, label %end

end:
  br label %entry2

entry2:
  %syncreg2 = tail call token @llvm.syncregion.start()
  %cmp6 = icmp sgt i64 %n, 0
  br i1 %cmp6, label %detach.preheader2, label %sync2

detach.preheader2:
  br label %detach2

detach2:
  %iv2 = phi i64 [ 0, %detach.preheader2 ], [ %iv.next2, %inc2 ]
  %iv.next2 = add nuw nsw i64 %iv2, 1
  detach within %syncreg2, label %body2, label %inc2

body2:
  %arrayidx2 = getelementptr inbounds i64, ptr %c, i64 %iv2
  store i64 %n, ptr %arrayidx2, align 4
  reattach within %syncreg2, label %inc2

inc2:
  %exitcond.not2 = icmp eq i64 %iv.next2, %n
  br i1 %exitcond.not2, label %sync2, label %detach2, !llvm.loop !3

sync2:
  sync within %syncreg2, label %end2

end2:
  ret void
}

declare token @llvm.syncregion.start() #1

attributes #0 = { nounwind memory(argmem: write) uwtable }
attributes #1 = { mustprogress nounwind willreturn memory(argmem: readwrite) }

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}
!3 = distinct !{!3, !4, !5, !6}
!4 = !{!"tapir.loop.spawn.strategy", i32 1}
!5 = !{!"llvm.loop.unroll.disable"}
!6 = !{!"tapir.loop.target", i32 4}
)m";

// There are no tapir loops. getRequiredTTs() should return an empty list.
TEST(TapirTargetAnalysisTest, noTapirLoops) {
  constexpr StringRef ir = R"m(
target triple = "x86_64-unknown-linux-gnu"

define void @f(ptr nocapture noundef writeonly %c, i64 noundef %n) #0 {
entry:
  %cmp5 = icmp sgt i64 %n, 0
  br i1 %cmp5, label %detach.preheader, label %sync

detach.preheader:
  br label %detach

detach:
  %iv = phi i64 [ 0, %detach.preheader ], [ %iv.next, %inc ]
  %iv.next = add nuw nsw i64 %iv, 1
  br label %body

body:
  %arrayidx = getelementptr inbounds i64, ptr %c, i64 %iv
  store i64 %n, ptr %arrayidx, align 4
  br label %inc

inc:
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %sync, label %detach, !llvm.loop !0

sync:
  br label %entry2

entry2:
  %cmp6 = icmp sgt i64 %n, 0
  br i1 %cmp6, label %detach.preheader2, label %sync2

detach.preheader2:
  br label %detach2

detach2:
  %iv2 = phi i64 [ 0, %detach.preheader2 ], [ %iv.next2, %inc2 ]
  %iv.next2 = add nuw nsw i64 %iv2, 1
  br label %body2

body2:
  %arrayidx2 = getelementptr inbounds i64, ptr %c, i64 %iv2
  store i64 %n, ptr %arrayidx2, align 4
  br label %inc2

inc2:
  %exitcond.not2 = icmp eq i64 %iv.next2, %n
  br i1 %exitcond.not2, label %sync2, label %detach2, !llvm.loop !0

sync2:
  ret void
}

attributes #0 = { nounwind memory(argmem: write) uwtable }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
)m";

  LLVMContext ctx;
  SMDiagnostic err;
  driver::KitsuneOptions kitOpts;
  std::optional<TapirTargetOptions> tto = std::nullopt;

  kitOpts.setTapirTarget(TTID::Serial);
  kitOpts.setCudaArch("sm_17");
  tto =
      TapirTargetOptions::create(kitOpts, OptznLevel::O2, FPOpFusion::Standard);

  std::unique_ptr<Module> m = parseAssemblyString(ir, err, ctx);
  Function *f = m->getFunction("f");

  FunctionAnalysisManager fam;
  fam.registerPass([&] { return DominatorTreeAnalysis(); });
  fam.registerPass([&] { return LoopAnalysis(); });
  fam.registerPass([&] { return PassInstrumentationAnalysis(); });
  fam.registerPass([&] { return TaskAnalysis(); });

  ModuleAnalysisManager mam;
  mam.registerPass([&] { return FunctionAnalysisManagerModuleProxy(fam); });
  mam.registerPass([&] { return PassInstrumentationAnalysis(); });

  auto tta = std::make_unique<TapirTargetAnalysis>(tto);
  TapirTargetInfo tgi = tta->run(*m, mam);

  EXPECT_TRUE(tgi.hasID());
  EXPECT_TRUE(tgi.getIDIfExists());
  EXPECT_EQ(*tgi.getIDIfExists(), TTID::Serial);
  EXPECT_EQ(tgi.getID(), TTID::Serial);
  EXPECT_EQ(tgi.getOptions().getCudaArch(), StringRef("sm_17"));
  EXPECT_EQ(tgi.getRequiredTTs(*f).size(), 0UL);
  EXPECT_EQ(tgi.getRequiredTTs(*m).size(), 0UL);
}

// None of the tapir loops have a tapir target set on them. getRequiredTTs()
// should only return the primary tapir target id.
TEST(TapirTargetAnalysisTest, noHints) {
  LLVMContext ctx;
  SMDiagnostic err;
  driver::KitsuneOptions kitOpts;
  std::optional<TapirTargetOptions> tto = std::nullopt;

  kitOpts.setTapirTarget(TTID::Serial);
  tto =
      TapirTargetOptions::create(kitOpts, OptznLevel::O2, FPOpFusion::Standard);

  std::unique_ptr<Module> m = parseAssemblyString(moduleNoHints, err, ctx);
  Function *f = m->getFunction("f");

  FunctionAnalysisManager fam;
  fam.registerPass([&] { return DominatorTreeAnalysis(); });
  fam.registerPass([&] { return LoopAnalysis(); });
  fam.registerPass([&] { return PassInstrumentationAnalysis(); });
  fam.registerPass([&] { return TaskAnalysis(); });

  ModuleAnalysisManager mam;
  mam.registerPass([&] { return FunctionAnalysisManagerModuleProxy(fam); });
  mam.registerPass([&] { return PassInstrumentationAnalysis(); });

  auto tta = std::make_unique<TapirTargetAnalysis>(tto);
  TapirTargetInfo tgi = tta->run(*m, mam);

  // The expected array will be in ascending order. If the order of tapir
  // targets or their numerical values are changed, this will need to be
  // updated.
  std::vector<TTID> expected = {TTID::Serial};
  EXPECT_EQ(tgi.getRequiredTTs(*f), expected);
  EXPECT_EQ(tgi.getRequiredTTs(*m), expected);
}

// One of the two tapir loops has a target set, the other does not.
// getRequiredTTs() should return the primary tapir target ID and the target on
// the loop.
TEST(TapirTargetAnalysisTest, withHintsMixed) {
  LLVMContext ctx;
  SMDiagnostic err;
  driver::KitsuneOptions kitOpts;
  std::optional<TapirTargetOptions> tto = std::nullopt;

  kitOpts.setTapirTarget(TTID::Serial);
  kitOpts.setCudaArch("sm_17");
  tto =
      TapirTargetOptions::create(kitOpts, OptznLevel::O2, FPOpFusion::Standard);

  std::unique_ptr<Module> m =
      parseAssemblyString(moduleWithHintsMixed, err, ctx);
  Function *f = m->getFunction("f");

  FunctionAnalysisManager fam;
  fam.registerPass([&] { return DominatorTreeAnalysis(); });
  fam.registerPass([&] { return LoopAnalysis(); });
  fam.registerPass([&] { return PassInstrumentationAnalysis(); });
  fam.registerPass([&] { return TaskAnalysis(); });

  ModuleAnalysisManager mam;
  mam.registerPass([&] { return FunctionAnalysisManagerModuleProxy(fam); });
  mam.registerPass([&] { return PassInstrumentationAnalysis(); });

  auto tta = std::make_unique<TapirTargetAnalysis>(tto);
  TapirTargetInfo tgi = tta->run(*m, mam);

  // The expected array will be in ascending order. If the order of tapir
  // targets or their numerical values are changed, this will need to be
  // updated.
  std::vector<TTID> expected = {TTID::Serial, TTID::Hip};
  EXPECT_EQ(tgi.getRequiredTTs(*f), expected);
  EXPECT_EQ(tgi.getRequiredTTs(*m), expected);
}

// All tapir loops have a tapir target set on them. getRequiredTTs() should not
// return the primary tapir target id.
TEST(TapirTargetAnalysisTest, withHintsNoDefault) {
  constexpr StringRef ir = R"m(
target triple = "x86_64-unknown-linux-gnu"

define void @f(ptr nocapture noundef writeonly %c, i64 noundef %n) #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp5 = icmp sgt i64 %n, 0
  br i1 %cmp5, label %detach.preheader, label %sync

detach.preheader:
  br label %detach

detach:
  %iv = phi i64 [ 0, %detach.preheader ], [ %iv.next, %inc ]
  %iv.next = add nuw nsw i64 %iv, 1
  detach within %syncreg, label %body, label %inc

body:
  %arrayidx = getelementptr inbounds i64, ptr %c, i64 %iv
  store i64 %n, ptr %arrayidx, align 4
  reattach within %syncreg, label %inc

inc:
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %sync, label %detach, !llvm.loop !0

sync:
  sync within %syncreg, label %end

end:
  br label %entry2

entry2:
  %syncreg2 = tail call token @llvm.syncregion.start()
  %cmp6 = icmp sgt i64 %n, 0
  br i1 %cmp6, label %detach.preheader2, label %sync2

detach.preheader2:
  br label %detach2

detach2:
  %iv2 = phi i64 [ 0, %detach.preheader2 ], [ %iv.next2, %inc2 ]
  %iv.next2 = add nuw nsw i64 %iv2, 1
  detach within %syncreg2, label %body2, label %inc2

body2:
  %arrayidx2 = getelementptr inbounds i64, ptr %c, i64 %iv2
  store i64 %n, ptr %arrayidx2, align 4
  reattach within %syncreg2, label %inc2

inc2:
  %exitcond.not2 = icmp eq i64 %iv.next2, %n
  br i1 %exitcond.not2, label %sync2, label %detach2, !llvm.loop !4

sync2:
  sync within %syncreg2, label %end2

end2:
  ret void
}

declare token @llvm.syncregion.start() #1

attributes #0 = { nounwind memory(argmem: write) uwtable }
attributes #1 = { mustprogress nounwind willreturn memory(argmem: readwrite) }

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}
!3 = !{!"tapir.loop.target", i32 4}
!4 = distinct !{!4, !1, !2, !5}
!5 = !{!"tapir.loop.target", i32 2}
)m";

  LLVMContext ctx;
  SMDiagnostic err;
  driver::KitsuneOptions kitOpts;
  std::optional<TapirTargetOptions> tto = std::nullopt;

  kitOpts.setTapirTarget(TTID::Serial);
  tto =
      TapirTargetOptions::create(kitOpts, OptznLevel::O2, FPOpFusion::Standard);

  std::unique_ptr<Module> m = parseAssemblyString(ir, err, ctx);
  Function *f = m->getFunction("f");

  FunctionAnalysisManager fam;
  fam.registerPass([&] { return DominatorTreeAnalysis(); });
  fam.registerPass([&] { return LoopAnalysis(); });
  fam.registerPass([&] { return PassInstrumentationAnalysis(); });
  fam.registerPass([&] { return TaskAnalysis(); });

  ModuleAnalysisManager mam;
  mam.registerPass([&] { return FunctionAnalysisManagerModuleProxy(fam); });
  mam.registerPass([&] { return PassInstrumentationAnalysis(); });

  auto tta = std::make_unique<TapirTargetAnalysis>(tto);
  TapirTargetInfo tgi = tta->run(*m, mam);

  // The expected array will be in ascending order. If the order of tapir
  // targets or their numerical values are changed, this will need to be
  // updated.
  std::vector<TTID> expected = {TTID::Cuda, TTID::Hip};
  EXPECT_EQ(tgi.getRequiredTTs(*f), expected);
  EXPECT_EQ(tgi.getRequiredTTs(*m), expected);
}

// Check that in a module with multiple functions, the required TT's are
// computed correctly for both the functions and the module.
TEST(TapirTargetAnalysisTest, withMultipleFuncs) {
  constexpr StringRef ir = R"m(
target triple = "x86_64-unknown-linux-gnu"

define void @f(ptr nocapture noundef writeonly %c, i64 noundef %n) #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp5 = icmp sgt i64 %n, 0
  br i1 %cmp5, label %detach.preheader, label %sync

detach.preheader:
  br label %detach

detach:
  %iv = phi i64 [ 0, %detach.preheader ], [ %iv.next, %inc ]
  %iv.next = add nuw nsw i64 %iv, 1
  detach within %syncreg, label %body, label %inc

body:
  %arrayidx = getelementptr inbounds i64, ptr %c, i64 %iv
  store i64 %n, ptr %arrayidx, align 4
  reattach within %syncreg, label %inc

inc:
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %sync, label %detach, !llvm.loop !0

sync:
  sync within %syncreg, label %end

end:
  ret void
}

define void @g(ptr nocapture noundef writeonly %c, i64 noundef %n) #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp6 = icmp sgt i64 %n, 0
  br i1 %cmp6, label %detach.preheader, label %sync

detach.preheader:
  br label %detach

detach:
  %iv = phi i64 [ 0, %detach.preheader ], [ %iv.next, %inc ]
  %iv.next = add nuw nsw i64 %iv, 1
  detach within %syncreg, label %body, label %inc

body:
  %arrayidx = getelementptr inbounds i64, ptr %c, i64 %iv
  store i64 %n, ptr %arrayidx, align 4
  reattach within %syncreg, label %inc

inc:
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %sync, label %detach, !llvm.loop !4

sync:
  sync within %syncreg, label %end

end:
  ret void
}

declare token @llvm.syncregion.start() #1

attributes #0 = { nounwind memory(argmem: write) uwtable }
attributes #1 = { mustprogress nounwind willreturn memory(argmem: readwrite) }

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}
!3 = !{!"tapir.loop.target", i32 2}
!4 = distinct !{!4, !1, !2, !5}
!5 = !{!"tapir.loop.target", i32 4}
)m";

  LLVMContext ctx;
  SMDiagnostic err;
  driver::KitsuneOptions kitOpts;
  std::optional<TapirTargetOptions> tto = std::nullopt;

  kitOpts.setTapirTarget(TTID::Serial);
  tto =
      TapirTargetOptions::create(kitOpts, OptznLevel::O2, FPOpFusion::Standard);

  std::unique_ptr<Module> m = parseAssemblyString(ir, err, ctx);
  Function *f = m->getFunction("f");
  Function *g = m->getFunction("g");

  FunctionAnalysisManager fam;
  fam.registerPass([&] { return DominatorTreeAnalysis(); });
  fam.registerPass([&] { return LoopAnalysis(); });
  fam.registerPass([&] { return PassInstrumentationAnalysis(); });
  fam.registerPass([&] { return TaskAnalysis(); });

  ModuleAnalysisManager mam;
  mam.registerPass([&] { return FunctionAnalysisManagerModuleProxy(fam); });
  mam.registerPass([&] { return PassInstrumentationAnalysis(); });

  auto tta = std::make_unique<TapirTargetAnalysis>(tto);
  TapirTargetInfo tgi = tta->run(*m, mam);

  // The expected array will be in ascending order. If the order of tapir
  // targets or their numerical values are changed, this will need to be
  // updated.
  std::vector<TTID> expectedF = {TTID::Cuda};
  std::vector<TTID> expectedG = {TTID::Hip};
  std::vector<TTID> expected = {TTID::Cuda, TTID::Hip};
  EXPECT_EQ(tgi.getRequiredTTs(*f), expectedF);
  EXPECT_EQ(tgi.getRequiredTTs(*g), expectedG);
  EXPECT_EQ(tgi.getRequiredTTs(*m), expected);
}

// If a tapir target options object has not been set, getRequiredTTs() will
// always return an empty vector.
TEST(TapirTargetAnalysisTest, noTTO) {
  LLVMContext ctx;
  SMDiagnostic err;
  std::optional<TapirTargetOptions> tto = std::nullopt;

  {
    std::unique_ptr<Module> m =
        parseAssemblyString(moduleWithHintsMixed, err, ctx);
    Function *f = m->getFunction("f");

    ModuleAnalysisManager mam;
    auto tta = std::make_unique<TapirTargetAnalysis>(std::nullopt);
    TapirTargetInfo tgi = tta->run(*m, mam);

    EXPECT_FALSE(tgi.hasID());
    EXPECT_FALSE(tgi.getIDIfExists());
    EXPECT_EQ(tgi.getRequiredTTs(*f).size(), 0UL);
  }

  {
    std::unique_ptr<Module> m = parseAssemblyString(moduleNoHints, err, ctx);
    Function *f = m->getFunction("f");

    ModuleAnalysisManager mam;
    auto tta = std::make_unique<TapirTargetAnalysis>(std::nullopt);
    TapirTargetInfo tgi = tta->run(*m, mam);

    EXPECT_FALSE(tgi.hasID());
    EXPECT_FALSE(tgi.getIDIfExists());
    EXPECT_EQ(tgi.getRequiredTTs(*f).size(), 0UL);
    EXPECT_EQ(tgi.getRequiredTTs(*m).size(), 0UL);
  }
}

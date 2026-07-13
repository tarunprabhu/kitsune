//===- TTObjectsAnalysisTest.cpp - TTObjectsAnalysis tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Analysis/TTObjectsAnalysis.h"
#include "kitsune/Core/KitOptions.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

class KitTTObjectsAnalysis : public ::testing::Test {
protected:
  LLVMContext ctx;
  SMDiagnostic err;
  std::unique_ptr<Module> m;
  std::unique_ptr<TTObjectsAnalysis> tta;
  std::optional<TTOptions> tto;
  FunctionAnalysisManager fam;
  ModuleAnalysisManager mam;
  Function *f = nullptr;
  Function *g = nullptr;

public:
  TTObjects setup(StringRef ll, const KitOptions &kitOpts) {
    m = parseAssemblyString(ll, err, ctx);
    f = m->getFunction("f");
    g = m->getFunction("g");

    FunctionAnalysisManager fam;
    fam.registerPass([&] { return DominatorTreeAnalysis(); });
    fam.registerPass([&] { return LoopAnalysis(); });
    fam.registerPass([&] { return PassInstrumentationAnalysis(); });
    fam.registerPass([&] { return TaskAnalysis(); });

    ModuleAnalysisManager mam;
    mam.registerPass([&] { return FunctionAnalysisManagerModuleProxy(fam); });
    mam.registerPass([&] { return PassInstrumentationAnalysis(); });

    tto = TTOptions::create(kitOpts, OptznLevel::O2, FPOpFusion::Standard);
    tta = std::make_unique<TTObjectsAnalysis>(tto);

    return tta->run(*m, mam);
  }
};

static constexpr StringRef noTapirLoops = R"m(
define void @f(i64 %n) {
entry:
  br label %for.i

for.i:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %for.i ]
  %i.inc = add i64 %i, 1
  %i.cmp = icmp eq i64 %i.inc, %n
  br i1 %i.cmp, label %for.i.exit, label %for.i, !llvm.loop !0

for.i.exit:
  br label %for.j

for.j:
  %j = phi i64 [ 0, %for.i.exit ], [ %j.inc, %for.j ]
  %j.inc = add i64 %j, 1
  %j.cmp = icmp eq i64 %j.inc, %n
  br i1 %j.cmp, label %for.j.exit, label %for.j, !llvm.loop !1

for.j.exit:
  ret void
}

!0 = distinct !{!0}
!1 = distinct !{!1}
)m";

static constexpr StringRef mixed1 = R"m(
define void @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %detach

detach:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %inc ]
  detach within %syncreg, label %body, label %inc

body:
  reattach within %syncreg, label %inc

inc:
  %iv.next = add i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %sync, label %detach, !llvm.loop !0

sync:
  sync within %syncreg, label %entry2

entry2:
  %syncreg2 = tail call token @llvm.syncregion.start()
  br label %detach2

detach2:
  %iv2 = phi i64 [ 0, %entry2 ], [ %iv.next2, %inc2 ]
  detach within %syncreg2, label %body2, label %inc2

body2:
  reattach within %syncreg2, label %inc2

inc2:
  %iv.next2 = add i64 %iv2, 1
  %exitcond.not2 = icmp eq i64 %iv.next2, %n
  br i1 %exitcond.not2, label %sync2, label %detach2, !llvm.loop !2

sync2:
  sync within %syncreg2, label %end2

end2:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 1}
!2 = distinct !{!2, !3}
!3 = !{!"tapir.loop.target", i32 1024}
)m";

static constexpr StringRef mixed2 = R"m(
define void @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %detach

detach:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %inc ]
  detach within %syncreg, label %body, label %inc

body:
  reattach within %syncreg, label %inc

inc:
  %iv.next = add i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %sync, label %detach, !llvm.loop !0

sync:
  sync within %syncreg, label %end

end:
  ret void
}

define void @g(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %detach

detach:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %inc ]
  detach within %syncreg, label %body, label %inc

body:
  reattach within %syncreg, label %inc

inc:
  %iv.next = add i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %sync, label %detach, !llvm.loop !2

sync:
  sync within %syncreg, label %end

end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 1}
!2 = distinct !{!2, !3}
!3 = !{!"tapir.loop.target", i32 1024}
)m";

// There are no tapir loops. getRequiredTTs() should return an empty list. Even
// so, a tapir target for the prrimary tapir target must have been created.
TEST_F(KitTTObjectsAnalysis, noTapirLoops) {
  driver::KitOptions kitOpts;
  kitOpts.setTTID(TTID::Serial);
  kitOpts.setCudaArch("sm_17");

  TTObjects ttObjs = setup(noTapirLoops, kitOpts);

  EXPECT_TRUE(ttObjs.hasTTID());
  EXPECT_TRUE(ttObjs.getTTIDOrNull());
  EXPECT_EQ(*ttObjs.getTTIDOrNull(), TTID::Serial);
  EXPECT_EQ(ttObjs.getTTID(), TTID::Serial);
  EXPECT_EQ(ttObjs.getOptions().getCudaArch(), StringRef("sm_17"));
  EXPECT_EQ(ttObjs.getRequiredTTs(*f).size(), 0UL);
  EXPECT_EQ(ttObjs.getRequiredTTs(*m).size(), 0UL);

  EXPECT_TRUE(ttObjs.hasTT(TTID::Serial));
}

// If a tapir target options object has not been set, getRequiredTTs() will
// always return an empty vector.
TEST_F(KitTTObjectsAnalysis, noTTO) {
  KitOptions kitOpts;
  TTObjects ttObjs = setup(mixed1, kitOpts);

  EXPECT_FALSE(ttObjs.hasTTID());
  EXPECT_FALSE(ttObjs.getTTIDOrNull());
  EXPECT_EQ(ttObjs.getRequiredTTs(*f).size(), 0UL);
  EXPECT_EQ(ttObjs.getRequiredTTs(*m).size(), 0UL);
}

// All tapir loops must have a tapir target set. getRequiredTTs() should not
// return the primary tapir target id. TapirTarget objects should have been
// created for each of the id's.
TEST_F(KitTTObjectsAnalysis, mixed) {
  driver::KitOptions kitOpts;
  kitOpts.setTTID(TTID::Serial);

  TTObjects ttObjs = setup(mixed1, kitOpts);

  // The expected array will be in ascending order of the integer values of the
  // TTID's. These are unlikely to change.
  TTID expected[] = {TTID::Serial, TTID::Pthreads};
  EXPECT_EQ(ttObjs.getRequiredTTs(*f), ArrayRef(expected));
  EXPECT_EQ(ttObjs.getRequiredTTs(*m), ArrayRef(expected));

  EXPECT_TRUE(ttObjs.hasTT(TTID::Serial));
  EXPECT_TRUE(ttObjs.hasTT(TTID::Pthreads));
  EXPECT_FALSE(ttObjs.hasTT(TTID::Cuda));
  EXPECT_FALSE(ttObjs.hasTT(TTID::Hip));
}

// Check that, in a module with multiple functions, the required TT's are
// computed correctly for both the functions and the module. In each case, a
// TapirTarget object should have been created. A tapir target for the primary
// tapir target will also have been created unconditionally.
TEST_F(KitTTObjectsAnalysis, withMultipleFuncs) {
  driver::KitOptions kitOpts;
  kitOpts.setTTID(TTID::OpenMP);

  TTObjects ttObjs = setup(mixed2, kitOpts);

  // The expected array will be in ascending order of the integer values of the
  // TTID's. These are unlikely to change.
  TTID expectedF[] = {TTID::Serial};
  TTID expectedG[] = {TTID::Pthreads};
  TTID expected[] = {TTID::Serial, TTID::Pthreads};
  EXPECT_EQ(ttObjs.getRequiredTTs(*f), ArrayRef(expectedF));
  EXPECT_EQ(ttObjs.getRequiredTTs(*g), ArrayRef(expectedG));
  EXPECT_EQ(ttObjs.getRequiredTTs(*m), ArrayRef(expected));

  EXPECT_TRUE(ttObjs.hasTT(TTID::OpenMP));
  EXPECT_TRUE(ttObjs.hasTT(TTID::Serial));
  EXPECT_TRUE(ttObjs.hasTT(TTID::Pthreads));
  EXPECT_FALSE(ttObjs.hasTT(TTID::Cuda));
  EXPECT_FALSE(ttObjs.hasTT(TTID::Hip));
}

static constexpr StringRef mixedIntrsOnly = R"m(
declare void @thrdfn(i64, i64, ptr)

define void @f(i64 %n) {
  call void @llvm.kit.cpu.threads.launch(i32 512, ptr @thrdfn, i64 0, i64 %n, ptr null)
  ret void
}

define void @g(ptr addrspace(67) %buf, i64 %n) {
  call void @llvm.kit.mobile.free(i32 1024, ptr addrspace(67) %buf)
  ret void
}
)m";

// The getRequiredTTs must also work as expected when Kitsune's intrinsics are
// used, even if there are no tapir loops.
TEST_F(KitTTObjectsAnalysis, intrsOnly) {
  driver::KitOptions kitOpts;
  kitOpts.setTTID(TTID::Serial);

  TTObjects ttObjs = setup(mixedIntrsOnly, kitOpts);

  // The expected array will be in ascending order of the integer values of the
  // TTID's. These are unlikely to change.
  TTID expectedF[] = {TTID::OpenMP};
  TTID expectedG[] = {TTID::Pthreads};
  TTID expectedM[] = {TTID::OpenMP, TTID::Pthreads};
  EXPECT_EQ(ttObjs.getRequiredTTs(*f), ArrayRef(expectedF));
  EXPECT_EQ(ttObjs.getRequiredTTs(*g), ArrayRef(expectedG));
  EXPECT_EQ(ttObjs.getRequiredTTs(*m), ArrayRef(expectedM));
}

static constexpr StringRef mixedIntrsLoops = R"m(
declare void @thrdfn(i64, i64, ptr)

define void @f(i64 %n) {
  call void @llvm.kit.cpu.threads.launch(i32 512, ptr @thrdfn, i64 0, i64 %n, ptr null)
  ret void
}

define void @g(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %detach

detach:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %inc ]
  detach within %syncreg, label %body, label %inc

body:
  reattach within %syncreg, label %inc

inc:
  %iv.next = add i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %sync, label %detach, !llvm.loop !0

sync:
  sync within %syncreg, label %end

end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 1024}
)m";

// The getRequiredTTs must also work as expected when Kitsune's intrinsics are
// used, even if there are no tapir loops.
TEST_F(KitTTObjectsAnalysis, intrsLoops) {
  driver::KitOptions kitOpts;
  kitOpts.setTTID(TTID::Serial);

  TTObjects ttObjs = setup(mixedIntrsLoops, kitOpts);

  // The expected array will be in ascending order of the integer values of the
  // TTID's. These are unlikely to change.
  TTID expectedF[] = {TTID::OpenMP};
  TTID expectedG[] = {TTID::Pthreads};
  TTID expectedM[] = {TTID::OpenMP, TTID::Pthreads};
  EXPECT_EQ(ttObjs.getRequiredTTs(*f), ArrayRef(expectedF));
  EXPECT_EQ(ttObjs.getRequiredTTs(*g), ArrayRef(expectedG));
  EXPECT_EQ(ttObjs.getRequiredTTs(*m), ArrayRef(expectedM));
}

} // namespace

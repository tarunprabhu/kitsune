; ------------------------------------------------------------------------------
; Check that specifying explicit optimization levels produces an appropriate
; pipeline. This does not attempt to be very thorough. It simply checks that
; the various optimization levels produce a "reasonably different" pipelines.
;
; The Kitsune post-tapir passes should not be run, but the mandatory passes
; should always be run.
;
; ------------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda --tapir-cuda-runtime-bc=%S/input/libdevice.ll %s \
; RUN:     -passes='tapir-lowering<O1>,emb-optimize' -o /dev/null \
; RUN:     -emb-print-pipeline-passes -emb-O0 \
; RUN:     | FileCheck %s --check-prefix=O0
;
; O0: NVVMReflectPass
; O0-NOT: LoopUnrollPass
; O0-NOT: EmbResolveLibDeviceCalls
; O0-NOT: GenerateCtors
; O0-SAME: GlobalDCEPass
; O0-SAME: VerifierPass
;
; ------------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda --tapir-cuda-runtime-bc=%S/input/libdevice.ll %s \
; RUN:     -passes='tapir-lowering<O1>,emb-optimize' -o /dev/null \
; RUN:     -emb-print-pipeline-passes -emb-O1 \
; RUN:     | FileCheck %s --check-prefix=O1
;
; O1: NVVMReflectPass
; O1-SAME: LoopUnrollPass<O1>
; O1-NOT: EmbResolveLibDeviceCalls
; O1-NOT: GenerateCtors
; O1-SAME: GlobalDCEPass
; O1-SAME: VerifierPass
;
; ------------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda --tapir-cuda-runtime-bc=%S/input/libdevice.ll %s \
; RUN:     -passes='tapir-lowering<O1>,emb-optimize' -o /dev/null \
; RUN:     -emb-print-pipeline-passes -emb-O2 \
; RUN:     | FileCheck %s --check-prefix=O2
;
; O2: NVVMReflectPass
; O2-SAME: LoopUnrollPass<O2>
; O2-NOT: EmbResolveLibDeviceCalls
; O2-NOT: GenerateCtors
; O2-SAME: GlobalDCEPass
; O2-SAME: VerifierPass
;
; ------------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda --tapir-cuda-runtime-bc=%S/input/libdevice.ll %s \
; RUN:     -passes='tapir-lowering<O1>,emb-optimize' -o /dev/null \
; RUN:     -emb-print-pipeline-passes -emb-O3 \
; RUN:     | FileCheck %s --check-prefix=O3
;
; O3: NVVMReflectPass
; O3-SAME: LoopUnrollPass<O3>
; O3-NOT: EmbResolveLibDeviceCalls
; O3-NOT: GenerateCtors
; O3-SAME: GlobalDCEPass
; O3-SAME: VerifierPass
;
; ------------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda --tapir-cuda-runtime-bc=%S/input/libdevice.ll %s \
; RUN:     -passes='tapir-lowering<O1>,emb-optimize' -o /dev/null \
; RUN:     -emb-print-pipeline-passes -emb-Os \
; RUN:     | FileCheck %s --check-prefix=Os
;
; Os: NVVMReflectPass
; Os-SAME: LoopRotatePass<header-duplication;no-prepare-for-lto>
; Os-NOT: LibCallsShrinkWrapPass
; Os-NOT: EmbResolveLibDeviceCalls
; Os-NOT: GenerateCtors
; Os-SAME: GlobalDCEPass
; Os-SAME: VerifierPass
;
; ------------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda --tapir-cuda-runtime-bc=%S/input/libdevice.ll %s \
; RUN:     -passes='tapir-lowering<O1>,emb-optimize' -o /dev/null \
; RUN:     -emb-print-pipeline-passes -emb-Oz \
; RUN:     | FileCheck %s --check-prefix=Oz
;
; Oz: NVVMReflectPass
; Oz-SAME: LoopRotatePass<no-header-duplication;no-prepare-for-lto>
; Oz-NOT: LibCallsShrinkWrapPass
; Oz-NOT: EmbResolveLibDeviceCalls
; Oz-NOT: GenerateCtors
; Oz-SAME: GlobalDCEPass
; Oz-SAME: VerifierPass
;
; ------------------------------------------------------------------------------

target triple = "x86_64-pc-linux-gnu"

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp5 = icmp sgt i64 %n, 0
  br i1 %cmp5, label %preheader, label %forall.sync

preheader:
  br label %forall.detach

forall.detach:
  %indvars.iv = phi i64 [ 0, %preheader ], [ %indvars.iv.next, %forall.inc ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  store i64 %n, ptr %arrayidx, align 4
  reattach within %syncreg, label %forall.inc

forall.inc:
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !0

forall.sync:
  sync within %syncreg, label %forall.end

forall.end:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 2}
!2 = !{!"llvm.loop.unroll.disable"}

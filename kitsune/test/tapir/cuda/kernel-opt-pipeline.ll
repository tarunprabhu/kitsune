; ------------------------------------------------------------------------------
; Check that specifying explicit optimization levels produces an appropriate
; pipeline. This does not attempt to be very thorough. It simply checks that
; the various optimization levels produce a "reasonably different" pipelines.
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
;
; ------------------------------------------------------------------------------

target triple = "x86_64-pc-linux-gnu"

; Function Attrs: nounwind memory(argmem: write) uwtable
define dso_local void @f(ptr nocapture noundef writeonly %c, i64 noundef %n) #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp5 = icmp sgt i64 %n, 0
  br i1 %cmp5, label %forall.detach.preheader, label %forall.sync

forall.detach.preheader:                          ; preds = %entry
  br label %forall.detach

forall.detach:                                    ; preds = %forall.detach.preheader, %forall.inc
  %indvars.iv = phi i64 [ 0, %forall.detach.preheader ], [ %indvars.iv.next, %forall.inc ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:                                      ; preds = %forall.detach
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  store i64 %n, ptr %arrayidx, align 4
  reattach within %syncreg, label %forall.inc

forall.inc:                                       ; preds = %forall.body, %forall.detach
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !0

forall.sync:                                      ; preds = %forall.inc, %entry
  sync within %syncreg, label %forall.end

forall.end:                                       ; preds = %forall.sync
  ret void
}

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare token @llvm.syncregion.start() #1

attributes #0 = { nounwind memory(argmem: write) uwtable }
attributes #1 = { mustprogress nounwind willreturn memory(argmem: readwrite) }

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}

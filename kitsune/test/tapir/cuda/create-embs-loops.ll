; If there are tapir loops in the module, create a global variable for embedded
; bitcode, even if there are no device functions.
;
; RUN: mkdir -p %t
; RUN: opt --tapir=cuda -passes=kit-embs,verify -S -o %t/loops.ll %s
; RUN: cat %t/loops.ll \
; RUN:     | FileCheck %s -check-prefix HOST
; RUN: cat %t/loops.ll | kit-mbc -S -o - \
; RUN:     | FileCheck %s -check-prefix DEVICE
;
; HOST: @{{.+}} = {{.*}}constant [{{[0-9]+}} x i8] c"BC{{.+}}"
; HOST-SAME: #[[BC:[0-9]+]]
; HOST: @{{.+}} = {{.*}}constant [0 x i8] zeroinitializer
; HOST-SAME: #[[FB:[0-9]+]]
; HOST-DAG: #[[BC]] = { kit_bc kit_tt(2) }
; HOST-DAG: #[[FB]] = { kit_fb kit_tt(2) }
;
; At this time, we do not clone any global values called by the tapir loops in
; the module, so the device module should be empty with the exception of the
; required metadata nodes.
;
; DEVICE: ModuleID
; DEVICE-NEXT: target datalayout
; DEVICE-NEXT: target triple = "nvptx64-nvidia-cuda"
; DEVICE-EMPTY:
; DEVICE-NEXT: !kitsune.device.module.flags

define void @callee(i64 %i) {
  ret void
}

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
  call void @callee(i64 %indvars.iv)
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
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}

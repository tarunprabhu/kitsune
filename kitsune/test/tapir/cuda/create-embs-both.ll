; If both device functions and tapir loops exist, only the device functions
; (and reachable global values) should be cloned into the embedded module. Any
; global values reachable from the tapir loops should not be cloned.
;
; NOTE: This may change in the future, in which case, this should comment should
; be removed and the comment above this should be updated.
;
; RUN: mkdir -p %t
; RUN: opt --tapir=cuda -passes=kit-embs,verify -S -o %t/both.ll %s
; RUN: cat %t/both.ll \
; RUN:     | FileCheck %s -check-prefix HOST
; RUN: cat %t/both.ll | kit-mbc -S -o - \
; RUN:     | FileCheck %s -check-prefix DEVICE
;
; HOST: @{{.+}} = {{.*}}constant [{{[0-9]+}} x i8] c"BC{{.+}}"
; HOST-SAME: #[[BC:[0-9]+]]
; HOST: @__nv_fatbin = external constant [0 x i8], section ".nv_fatbin"
; HOST-SAME: #[[FB:[0-9]+]]
; HOST-DAG: #[[BC]] = { kit_bc kit_tt(2) }
; HOST-DAG: #[[FB]] = { kit_fb kit_tt(2) }
;
; DEVICE-NOT: @gd2
; DEVICE-NOT: @h4
; DEVICE-NOT: @d2
; DEVICE-NOT: @h4
; DEVICE-DAG: @gd1
; DEVICE-DAG: @d1
; DEVICE-DAG: @d3

@gd1 = global i64 203
@gd2 = global i64 209
@gh = global i32 382

define void @d1() #0 {
  call void @d3(ptr @gd1)
  ret void
}

define void @d2(ptr %p) {
  ret void
}

define void @d3(ptr %p) {
  ret void
}

define void @h4() {
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
  call void @d2(ptr @gd2)
  reattach within %syncreg, label %forall.inc

forall.inc:
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !0

forall.sync:
  sync within %syncreg, label %forall.end

forall.end:
  ret void
}

attributes #0 = { kit_device }

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}

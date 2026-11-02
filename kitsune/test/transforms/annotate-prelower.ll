; Check that tapir loops intended to be executed on a CPU are annotated
; correctly.
;
; RUN: opt -passes="kit-annotate-prelower" -S %s | FileCheck %s

; CHECK-LABEL: @ps
; CHECK: !llvm.loop ![[PS_J:[0-9]+]]
; CHECK: !llvm.loop ![[PS_I:[0-9]+]]
;
; forall (i ...)
;   for (j ...)
;     ;
define void @ps(i64 %m, i64 %n) {
entry:
  %sr.i = tail call token @llvm.syncregion.start()
  br label %header.i

header.i:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch.i ]
  detach within %sr.i, label %body.i, label %latch.i

body.i:
  %sr.j = tail call token @llvm.syncregion.start()
  br label %header.j

header.j:
  %j = phi i64 [ 0, %body.i ], [ %inc.j, %header.j ]
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %reattach.i, label %header.j, !llvm.loop !1

reattach.i:
  reattach within %sr.i, label %latch.i

latch.i:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %sync.i, label %header.i, !llvm.loop !2

sync.i:
  sync within %sr.i, label %exit

exit:
  ret void
}

; CHECK-LABEL: @sp
; CHECK: !llvm.loop ![[SP_J:[0-9]+]]
; CHECK: !llvm.loop ![[SP_I:[0-9]+]]
;
; for (i ...)
;   forall (j ...)
;     ;
define void @sp(i64 %m, i64 %n) {
entry:
  br label %header.i

header.i:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch.i ]
  br label %body.i

body.i:
  %sr.j = tail call token @llvm.syncregion.start()
  br label %header.j

header.j:
  %j = phi i64 [ 0, %body.i ], [ %inc.j, %latch.j ]
  detach within %sr.j, label %body.j, label %latch.j

body.j:
  reattach within %sr.j, label %latch.j

latch.j:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %sync.j, label %header.j, !llvm.loop !3

sync.j:
  sync within %sr.j, label %latch.i

latch.i:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %exit, label %header.i, !llvm.loop !4

exit:
  ret void
}

;-------------------------------------------------------------------------------
;
; CHECK-DAG: ![[TARGET:[0-9]+]] = !{!"tapir.loop.target", i32 512}
; CHECK-DAG: ![[LOWER:[0-9]+]] = !{!"tapir.loop.lowering.enabled"}
;
; ------------------------------------------------------------------------------
;
; forall (i ...)
;   for (j ...)
;     ;
;
; CHECK-DAG: ![[PS_J]] = distinct !{![[PS_J]]}
; CHECK-DAG: ![[PS_I]] = distinct !{![[PS_I]], ![[TARGET]], ![[LOWER]]}
;
;-------------------------------------------------------------------------------
;
; for (i ...)
;   forall (j ...)
;     ...
;
; CHECK-DAG: ![[SP_J]] = distinct !{![[SP_J]], ![[TARGET]], ![[LOWER]]}
; CHECK-DAG: ![[SP_I]] = distinct !{![[SP_I]]}
;
;-------------------------------------------------------------------------------

!0 = !{!"tapir.loop.target", i32 512}
!1 = distinct !{!1}
!2 = distinct !{!2, !0}
!3 = distinct !{!3, !0}
!4 = distinct !{!4}

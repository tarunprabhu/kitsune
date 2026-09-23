; The kit-annotate-early pass must only be run once. Since the function in this
; test has an attribute indicating that the pass has been run on it, the tapir
; loops in it will not be annotated, even though they should be.
;
; RUN: opt -passes="kit-annotate-early" -S %s | FileCheck %s

; CHECK-LABEL: @pp
; CHECK: llvm.loop ![[PP_J:[0-9]+]]
; CHECK: llvm.loop ![[PP_I:[0-9]+]]
;
; forall (i ...)
;   forall (j ...)
;
define void @pp(i64 %m, i64 %n) !kit.func !3 {
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
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !2

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

!0 = distinct !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}
!2 = distinct !{!2, !0}
!3 = distinct !{!3, !4}
!4 = !{!"kit.func.early.annotate.pass"}

; CHECK-DAG: ![[TARGET:[0-9]+]] = distinct !{!"tapir.loop.target", i32 1}
; CHECK-DAG: ![[PP_J]] = distinct !{![[PP_J]], ![[TARGET]]}
; CHECK-DAG: ![[PP_I]] = distinct !{![[PP_I]], ![[TARGET]]}

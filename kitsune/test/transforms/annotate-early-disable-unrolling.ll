; Check that the kit-annotate-early pass adds the llvm.loop.unroll.disable
; annotation to all tapir loops, but not to non-tapir loops.
;
; RUN: opt -passes="kit-annotate-early" -S %s \
; RUN:     | FileCheck %s

; CHECK-LABEL: @s
; CHECK: llvm.loop ![[S_J:[0-9]+]]
; CHECK: llvm.loop ![[S_I:[0-9]+]]
;
; for (i ...)
;   for (j ...)
;
define void @s(i64 %n) {
entry:
  %cmp.not = icmp eq i64 %n, 0
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i64 [ 0, %for.i.header ], [ %inc.j, %for.j ]
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.i.latch, label %for.j, !llvm.loop !2

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !1

for.i.exit:
  ret void
}

; CHECK-LABEL: @pp
; CHECK: llvm.loop ![[PP_J:[0-9]+]]
; CHECK: llvm.loop ![[PP_I:[0-9]+]]
;
; forall (i ...)
;   forall (j ...)
;
define void @pp(i64 %m, i64 %n) {
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
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !4

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !3

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

; CHECK-LABEL: @ps
; CHECK: llvm.loop ![[PS_J:[0-9]+]]
; CHECK: llvm.loop ![[PS_I:[0-9]+]]
;
; forall (i ...)
;   for (j ...)
define void @ps(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
  br label %for.j

for.j:
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j ]
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j, !llvm.loop !6

for.j.exit:
  reattach within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.sync, label %for.i.header, !llvm.loop !5

for.i.sync:
  sync within %syncreg, label %for.i.exit

for.i.exit:
  ret void
}

; CHECK-LABEL: @sp
; CHECK: llvm.loop ![[SP_J:[0-9]+]]
; CHECK: llvm.loop ![[SP_I:[0-9]+]]
;
; for (i ...)
;   forall (j ...)
define void @sp(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.header ], [ %inc.j, %for.j.latch ]
  detach within %syncreg, label %for.j.body, label %for.j.latch

for.j.body:
  reattach within %syncreg, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.sync, label %for.j.header, !llvm.loop !8

for.j.sync:
  sync within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !7

for.i.exit:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1}
!2 = distinct !{!2}
!3 = distinct !{!3, !0}
!4 = distinct !{!4, !0}
!5 = distinct !{!5, !0}
!6 = distinct !{!6}
!7 = distinct !{!7}
!8 = distinct !{!8, !0}

;-------------------------------------------------------------------------------
;
; CHECK-DAG: ![[TARGET:[0-9]+]] = !{!"tapir.loop.target", i32 1}
; CHECK-DAG: ![[NOUNROLL:[0-9]+]] = !{!"llvm.loop.unroll.disable"}
;
;-------------------------------------------------------------------------------
;
; for (i ...)
;   for (j ...)
;
; CHECK-DAG: ![[S_I]] = distinct !{![[S_I]]}
; CHECK-DAG: ![[S_J]] = distinct !{![[S_J]]}
;
;-------------------------------------------------------------------------------
;
; forall (i ...)
;   forall (j ...)
;
; CHECK-DAG: ![[PP_I]] = distinct !{![[PP_I]], ![[TARGET]], ![[NOUNROLL]]}
; CHECK-DAG: ![[PP_J]] = distinct !{![[PP_J]], ![[TARGET]], ![[NOUNROLL]]}
;
;-------------------------------------------------------------------------------
;
; forall (i ...)
;   for (j ...)
;
; CHECK-DAG: ![[PS_I]] = distinct !{![[PS_I]], ![[TARGET]], ![[NOUNROLL]]}
; CHECK-DAG: ![[PS_J]] = distinct !{![[PS_J]]}
;
;-------------------------------------------------------------------------------
;
; for (i ...)
;   forall (j ...)
;
; CHECK-DAG: ![[SP_I]] = distinct !{![[SP_I]]}
; CHECK-DAG: ![[SP_J]] = distinct !{![[SP_J]], ![[TARGET]], ![[NOUNROLL]]}


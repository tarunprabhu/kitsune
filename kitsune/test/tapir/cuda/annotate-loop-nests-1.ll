; Check that the tapir loop annotator pass annotates loops correctly. Every
; function here contains a single loop nest. Each loop nest will contain exactly
; one tapir loop. The loop nest may contain other non-tapir loops.
;
; RUN: opt -passes="kit-annotate-prelower" -S %s \
; RUN:     | FileCheck %s

; CHECK-LABEL: @p
; CHECK: llvm.loop ![[P:[0-9]+]]
;
; forall (i ...)
define void @p(i64 %n) {
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
  br i1 %cmp.i, label %for.i.sync, label %for.i.header, !llvm.loop !0

for.i.sync:
  sync within %syncreg, label %for.i.exit

for.i.exit:
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
  br i1 %cmp.j, label %for.j.exit, label %for.j, !llvm.loop !2

for.j.exit:
  reattach within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.sync, label %for.i.header, !llvm.loop !3

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
  br i1 %cmp.j, label %for.j.sync, label %for.j.header, !llvm.loop !4

for.j.sync:
  sync within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !5

for.i.exit:
  ret void
}

;-------------------------------------------------------------------------------
;
; CHECK-DAG: ![[TARGET:[0-9]+]] = !{!"tapir.loop.target", i32 2}
; CHECK-DAG: ![[D1:[0-9]+]] = !{!"tapir.loop.perfect.depth", i32 1}
; CHECK-DAG: ![[L1:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 1}
; CHECK-DAG: ![[LOWER:[0-9]+]] = !{!"tapir.loop.lowering.enabled"}
;
;-------------------------------------------------------------------------------
;
; forall (i ...)
;
; CHECK-DAG: ![[P]] = distinct !{![[P]], ![[TARGET]], ![[LOWER]], ![[D1]], ![[L1]]}
;
;-------------------------------------------------------------------------------
;
; forall (i ...)
;   for (j ...)
;
; CHECK-DAG: ![[PS_J]] = distinct !{![[PS_J]]}
; CHECK-DAG: ![[PS_I]] = distinct !{![[PS_I]], ![[TARGET]], ![[LOWER]], ![[D1]], ![[L1]]}
;
;-------------------------------------------------------------------------------
;
; for (i ...)
;   forall (j ...)
;
; CHECK-DAG: ![[SP_J]] = distinct !{![[SP_J]], ![[TARGET]], ![[LOWER]], ![[D1]], ![[L1]]}
; CHECK-DAG: ![[SP_I]] = distinct !{![[SP_I]]}
;
;-------------------------------------------------------------------------------

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 2}
!2 = distinct !{!2}
!3 = distinct !{!3, !1}
!4 = distinct !{!4, !1}
!5 = distinct !{!5}

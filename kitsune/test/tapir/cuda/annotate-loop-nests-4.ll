; Check that the tapir loop annotator pass annotates loops correctly. Every
; function here contains a single loop nest. Each nest will contain exactly
; four tapir loops. The loop nest may contain other, non-tapir loops.
;
; RUN: opt -passes="kit-annotate-prelower" -S %s \
; RUN:     | FileCheck %s

; CHECK-LABEL: @pppp
; CHECK: llvm.loop ![[PPPP_L:[0-9]+]]
; CHECK: llvm.loop ![[PPPP_K:[0-9]+]]
; CHECK: llvm.loop ![[PPPP_J:[0-9]+]]
; CHECK: llvm.loop ![[PPPP_I:[0-9]+]]
;
; forall (i ...)
;   forall (j ...)
;     forall (k ...)
;       forall (l ...)
define void @pppp(i64 %m, i64 %n, i64 %p, i64 %q) {
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
  %syncreg.k = tail call token @llvm.syncregion.start()
  br label %for.k.header

for.k.header:
  %k = phi i64 [ 0, %for.j.body ], [ %inc.k, %for.k.latch ]
  detach within %syncreg.k, label %for.k.body, label %for.k.latch

for.k.body:
  %syncreg.l = tail call token @llvm.syncregion.start()
  br label %for.l.header

for.l.header:
  %l = phi i64 [ 0, %for.k.body ], [ %inc.l, %for.l.latch ]
  detach within %syncreg.l, label %for.l.body, label %for.l.latch

for.l.body:
  reattach within %syncreg.l, label %for.l.latch

for.l.latch:
  %inc.l = add i64 %l, 1
  %cmp.l = icmp eq i64 %inc.l, %q
  br i1 %cmp.l, label %for.l.exit, label %for.l.header, !llvm.loop !0

for.l.exit:
  sync within %syncreg.l, label %for.l.end

for.l.end:
  reattach within %syncreg.k, label %for.k.latch

for.k.latch:
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !2

for.k.exit:
  sync within %syncreg.k, label %for.k.end

for.k.end:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !3

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !4

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

;-------------------------------------------------------------------------------
;
; CHECK-DAG: ![[TARGET:[0-9]+]] = !{!"tapir.loop.target", i32 2}
; CHECK-DAG: ![[D4:[0-9]+]] = !{!"tapir.loop.perfect.depth", i32 4}
; CHECK-DAG: ![[L1:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 1}
; CHECK-DAG: ![[L2:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 2}
; CHECK-DAG: ![[L3:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 3}
; CHECK-DAG: ![[L4:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 4}
; CHECK-DAG: ![[LOWER:[0-9]+]] = !{!"tapir.loop.lowering.enabled"}
;
;-------------------------------------------------------------------------------
;
; forall (i ...)
;   forall (j ...)
;     forall (k ...)
;       forall (l ...)
;
; CHECK-DAG: ![[PPPP_L]] = distinct !{![[PPPP_L]], ![[TARGET]], ![[L4]]}
; CHECK-DAG: ![[PPPP_K]] = distinct !{![[PPPP_K]], ![[TARGET]], ![[L3]]}
; CHECK-DAG: ![[PPPP_J]] = distinct !{![[PPPP_J]], ![[TARGET]], ![[L2]]}
; CHECK-DAG: ![[PPPP_I]] = distinct !{![[PPPP_I]], ![[TARGET]], ![[LOWER]], ![[D4]], ![[L1]]}
;
;-------------------------------------------------------------------------------

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 2}
!2 = distinct !{!2, !1}
!3 = distinct !{!3, !1}
!4 = distinct !{!4, !1}

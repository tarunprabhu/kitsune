; Check that the tapir loop annotator pass annotates loops correctly. Every
; function here contains a single loop nest. Each nest will contain exactly
; three tapir loops. The loop nest may contain other, non-tapir loops.
;
; RUN: opt -passes="kit-annotate-tapir-loops" -S %s \
; RUN:     | FileCheck %s

; CHECK-LABEL: @ppp
; CHECK: llvm.loop ![[PPP_K:[0-9]+]]
; CHECK: llvm.loop ![[PPP_J:[0-9]+]]
; CHECK: llvm.loop ![[PPP_I:[0-9]+]]
;
; forall (i ...)
;   forall (j ...)
;     forall (k ...)
;
define void @ppp(i64 %m, i64 %n, i64 %p) {
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
  %k = phi i64 [0, %for.j.body ], [ %inc.k, %for.k.latch ]
  detach within %syncreg.k, label %for.k.body, label %for.k.latch

for.k.body:
  reattach within %syncreg.k, label %for.k.latch

for.k.latch:
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !0

for.k.exit:
  sync within %syncreg.k, label %for.k.end

for.k.end:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !2

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

; CHECK-LABEL: @ppps
; CHECK: llvm.loop ![[PPPS_L:[0-9]+]]
; CHECK: llvm.loop ![[PPPS_K:[0-9]+]]
; CHECK: llvm.loop ![[PPPS_J:[0-9]+]]
; CHECK: llvm.loop ![[PPPS_I:[0-9]+]]
;
; forall (i ...)
;   forall (j ...)
;     forall (k ...)
;       for (l ...)
;
define void @ppps(i64 %m, i64 %n, i64 %p, i64 %q) {
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
  br label %for.l.header

for.l.header:
  %l = phi i64 [ 0, %for.k.body], [ %inc.l, %for.l.header ]
  %inc.l = add i64 %l, 1
  %cmp.l = icmp eq i64 %inc.l, %q
  br i1 %cmp.l, label %for.l.exit, label %for.l.header, !llvm.loop !4

for.l.exit:
  reattach within %syncreg.k, label %for.k.latch

for.k.latch:
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !5

for.k.exit:
  sync within %syncreg.k, label %for.k.end

for.k.end:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !6

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !7

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

; CHECK-LABEL: @ppsp
; CHECK: llvm.loop ![[PPSP_L:[0-9]+]]
; CHECK: llvm.loop ![[PPSP_K:[0-9]+]]
; CHECK: llvm.loop ![[PPSP_J:[0-9]+]]
; CHECK: llvm.loop ![[PPSP_I:[0-9]+]]
;
; forall (i ...)
;   forall (j ...)
;     for (k ...)
;       forall (l ...)
;
define void @ppsp(i64 %m, i64 %n, i64 %p, i64 %q) {
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
  br label %for.k.header

for.k.header:
  %k = phi i64 [ 0, %for.j.body ], [ %inc.k, %for.k.latch ]
  br label %for.k.body

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
  br i1 %cmp.l, label %for.l.exit, label %for.l.header, !llvm.loop !8

for.l.exit:
  sync within %syncreg.l, label %for.k.latch

for.k.latch:
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !9

for.k.exit:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !10

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !11

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

; CHECK-LABEL: @pspp
; CHECK: llvm.loop ![[PSPP_L:[0-9]+]]
; CHECK: llvm.loop ![[PSPP_K:[0-9]+]]
; CHECK: llvm.loop ![[PSPP_J:[0-9]+]]
; CHECK: llvm.loop ![[PSPP_I:[0-9]+]]
;
; forall (i ...)
;   for (j ...)
;     forall (k ...)
;       forall (l ...)
;
define void @pspp(i64 %m, i64 %n, i64 %p, i64 %q) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  %cmp.m.not = icmp eq i64 %m, 0
  %cmp.n.not = icmp eq i64 %n, 0
  %cmp.p.not = icmp eq i64 %p, 0
  %cmp.q.not = icmp eq i64 %q, 0
  br i1 %cmp.m.not, label %for.i.exit, label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  br i1 %cmp.n.not, label %for.j.exit, label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j.latch ]
  br label %for.j.body

for.j.body:
  %syncreg.k = tail call token @llvm.syncregion.start()
  br i1 %cmp.p.not, label %for.k.exit, label %for.k.header

for.k.header:
  %k = phi i64 [ 0, %for.j.body ], [ %inc.k, %for.k.latch ]
  detach within %syncreg.k, label %for.k.body, label %for.k.latch

for.k.body:
  %syncreg.l = tail call token @llvm.syncregion.start()
  br i1 %cmp.q.not, label %for.l.exit, label %for.l.header

for.l.header:
  %l = phi i64 [ 0, %for.k.body ], [ %inc.l, %for.l.latch ]
  detach within %syncreg.l, label %for.l.body, label %for.l.latch

for.l.body:
  reattach within %syncreg.l, label %for.l.latch

for.l.latch:
  %inc.l = add i64 %l, 1
  %cmp.l = icmp eq i64 %inc.l, %q
  br i1 %cmp.l, label %for.l.exit, label %for.l.header, !llvm.loop !12

for.l.exit:
  sync within %syncreg.l, label %for.l.end

for.l.end:
  reattach within %syncreg.k, label %for.k.latch

for.k.latch:
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !13

for.k.exit:
  sync within %syncreg.k, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !14

for.j.exit:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add nuw i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !15

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

;-------------------------------------------------------------------------------
;
; CHECK-DAG: ![[TARGET:[0-9]+]] = !{!"tapir.loop.target", i32 4}
; CHECK-DAG: ![[D1:[0-9]+]] = !{!"tapir.loop.perfect.depth", i32 1}
; CHECK-DAG: ![[D2:[0-9]+]] = !{!"tapir.loop.perfect.depth", i32 2}
; CHECK-DAG: ![[D3:[0-9]+]] = !{!"tapir.loop.perfect.depth", i32 3}
; CHECK-DAG: ![[L1:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 1}
; CHECK-DAG: ![[L2:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 2}
; CHECK-DAG: ![[L3:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 3}
; CHECK-DAG: ![[LOWER:[0-9]+]] = !{!"tapir.loop.lowering.enabled"}
;
;-------------------------------------------------------------------------------
;
; forall (i ...)
;   forall (j ...)
;     forall (k ...)
;
; CHECK-DAG: ![[PPP_K]] = distinct !{![[PPP_K]], ![[TARGET]], ![[L3]]}
; CHECK-DAG: ![[PPP_J]] = distinct !{![[PPP_J]], ![[TARGET]], ![[L2]]}
; CHECK-DAG: ![[PPP_I]] = distinct !{![[PPP_I]], ![[TARGET]], ![[LOWER]], ![[D3]], ![[L1]]}
;
;-------------------------------------------------------------------------------
;
; forall (i ...)
;   forall (j ...)
;     forall (k ...)
;       for (l ...)
;
; CHECK-DAG: ![[PPPS_L]] = distinct !{![[PPPS_L]]}
; CHECK-DAG: ![[PPPS_K]] = distinct !{![[PPPS_K]], ![[TARGET]], ![[L3]]}
; CHECK-DAG: ![[PPPS_J]] = distinct !{![[PPPS_J]], ![[TARGET]], ![[L2]]}
; CHECK-DAG: ![[PPPS_I]] = distinct !{![[PPPS_I]], ![[TARGET]], ![[LOWER]], ![[D3]], ![[L1]]}
;
;-------------------------------------------------------------------------------
;
; forall (i ...)
;   forall (j ...)
;     for (k ...)
;       forall (l ...)
;
; Here, the non-parallel for loop in the nest will result in the maximum perfect
; depth being 2. The innermost forall is not part of the loop nest since its
; parent is not a tapir loop. For that same reason, it is also not the root of
; a different tapir loop nest. Therefore, neither the depth, nor the level
; annotations will be added since those are only added to perfectly nested tapir
; loops.
;
; CHECK-DAG: ![[PPSP_L]] = distinct !{![[PPSP_L]], ![[TARGET]]}
; CHECK-DAG: ![[PPSP_K]] = distinct !{![[PPSP_K]]}
; CHECK-DAG: ![[PPSP_J]] = distinct !{![[PPSP_J]], ![[TARGET]], ![[L2]]}
; CHECK-DAG: ![[PPSP_I]] = distinct !{![[PPSP_I]], ![[TARGET]], ![[LOWER]], ![[D2]], ![[L1]]}
;
;-------------------------------------------------------------------------------
;
; forall (i ...)
;   for (j ...)
;     forall (k ...)
;       forall (l ...)
;
; Here, the non-parallel for loop in the nest will result in the maximum perfect
; depth being 1. The two innermost forall loops are not part of the loop nest
; since the parent of the forall loop at depth 3 is not a tapir loop. For that
; same reason, it is also not the root of a different tapir loop nest.
; Therefore, neither the depth, nor the level annotations will be added since
; those are only added to perfectly nested tapir loops.
;
; CHECK-DAG: ![[PSPP_L]] = distinct !{![[PSPP_L]], ![[TARGET]]}
; CHECK-DAG: ![[PSPP_K]] = distinct !{![[PSPP_K]], ![[TARGET]]}
; CHECK-DAG: ![[PSPP_J]] = distinct !{![[PSPP_J]]}
; CHECK-DAG: ![[PSPP_I]] = distinct !{![[PSPP_I]], ![[TARGET]], ![[LOWER]], ![[D1]], ![[L1]]}
;
;-------------------------------------------------------------------------------

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 4}
!2 = distinct !{!2, !1}
!3 = distinct !{!3, !1}
!4 = distinct !{!4}
!5 = distinct !{!5, !1}
!6 = distinct !{!6, !1}
!7 = distinct !{!7, !1}
!8 = distinct !{!8, !1}
!9 = distinct !{!9}
!10 = distinct !{!10, !1}
!11 = distinct !{!11, !1}
!12 = distinct !{!12, !1}
!13 = distinct !{!13, !1}
!14 = distinct !{!14}
!15 = distinct !{!15, !1}

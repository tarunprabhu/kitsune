; Check that the tapir loop annotator pass annotates loops correctly. Every
; function here contains a single loop nest. Each nest will contain exactly two
; tapir loops. The loop nest may contain other, non-tapir loops.
;
; RUN: opt -passes="kit-annotate-tapir-loops" -S %s \
; RUN:     | FileCheck %s

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
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !0

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

; CHECK-LABEL: @pps
; CHECK: llvm.loop ![[PPS_K:[0-9]+]]
; CHECK: llvm.loop ![[PPS_J:[0-9]+]]
; CHECK: llvm.loop ![[PPS_I:[0-9]+]]
;
; forall (i ...)
;   forall (j ...)
;     for (k ...)
;
define dso_local void @pps(i64 %m, i64 %n, i64 %p) {
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
  br label %for.k.body

for.k.body:
  %k = phi i64 [ 0, %for.j.body ], [ %inc.k, %for.k.body ]
  %inc.k = add nuw i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k.body, !llvm.loop !3

for.k.exit:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add nuw i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !4

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add nuw i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !5

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

; CHECK-LABEL: @psp
; CHECK: llvm.loop ![[PSP_K:[0-9]+]]
; CHECK: llvm.loop ![[PSP_J:[0-9]+]]
; CHECK: llvm.loop ![[PSP_I:[0-9]+]]
;
; forall (i ...)
;   for (j ...)
;     forall (k ...)
;
define dso_local void @psp(i64 %m, i64 %n, i64 %p) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.k = tail call token @llvm.syncregion.start()
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j.latch ]
  br label %for.k.header

for.k.header:
  %k = phi i64 [ 0, %for.j.header ], [ %inc.k, %for.k.latch ]
  detach within %syncreg.k, label %for.k.body, label %for.k.latch

for.k.body:
  reattach within %syncreg.k, label %for.k.latch

for.k.latch:
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !6

for.k.exit:
  sync within %syncreg.k, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !7

for.j.exit:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !8

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

;-------------------------------------------------------------------------------
;
; CHECK-DAG: ![[TARGET:[0-9]+]] = !{!"tapir.loop.target", i32 2}
; CHECK-DAG: ![[D1:[0-9]+]] = !{!"tapir.loop.perfect.depth", i32 1}
; CHECK-DAG: ![[D2:[0-9]+]] = !{!"tapir.loop.perfect.depth", i32 2}
; CHECK-DAG: ![[L1:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 1}
; CHECK-DAG: ![[L2:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 2}
; CHECK-DAG: ![[LOWER:[0-9]+]] = !{!"tapir.loop.lowering.enabled"}
;
;-------------------------------------------------------------------------------
;
; forall (i ...)
;   forall (j ...)
;
; CHECK-DAG: ![[PP_J]] = distinct !{![[PP_J]], ![[TARGET]], ![[L2]]}
; CHECK-DAG: ![[PP_I]] = distinct !{![[PP_I]], ![[TARGET]], ![[LOWER]], ![[D2]], ![[L1]]}
;
;-------------------------------------------------------------------------------
;
; forall (i ...)
;   forall (j ...)
;     for (k ...)
;
; CHECK-DAG: ![[PPS_K]] = distinct !{![[PPS_K]]}
; CHECK-DAG: ![[PPS_J]] = distinct !{![[PPS_J]], ![[TARGET]], ![[L2]]}
; CHECK-DAG: ![[PPS_I]] = distinct !{![[PPS_I]], ![[TARGET]], ![[LOWER]], ![[D2]], ![[L1]]}
;
;-------------------------------------------------------------------------------
;
; forall (i ...)
;   for (j ...)
;     forall (k ...)
;
; Here, the non-parallel for loop in the nest will result in the maximum perfect
; depth being 2. The innermost forall is not part of the loop nest since its
; parent is not a tapir loop. For that same reason, it is also not the root of
; a different tapir loop nest. Therefore, neither the depth, nor the level
; annotations will be added since those are only added to perfectly nested tapir
; loops.
;
; CHECK-DAG: ![[PSP_K]] = distinct !{![[PSP_K]], ![[TARGET]]}
; CHECK-DAG: ![[PSP_J]] = distinct !{![[PSP_J]]}
; CHECK-DAG: ![[PSP_I]] = distinct !{![[PSP_I]], ![[TARGET]], ![[LOWER]], ![[D1]], ![[L1]]}
;
;-------------------------------------------------------------------------------

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 2}
!2 = distinct !{!2, !1}
!3 = distinct !{!3}
!4 = distinct !{!4, !1}
!5 = distinct !{!5, !1}
!6 = distinct !{!6, !1}
!7 = distinct !{!7}
!8 = distinct !{!8, !1}

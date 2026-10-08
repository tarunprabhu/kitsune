; Check that the tapir loop annotator pass annotates loops correctly. Every
; function here contains a single loop nest. Each nest will contain exactly two
; tapir loops. The loop nest may contain other, non-tapir loops.
;
; RUN: opt -passes="kit-annotate-early" -S %s | FileCheck %s

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
define void @pps(i64 %m, i64 %n, i64 %p) {
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
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k.body, !llvm.loop !3

for.k.exit:
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
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !5

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

; CHECK-LABEL: @spp
;
; CHECK: llvm.loop ![[SPP_K:[0-9]+]]
; CHECK: llvm.loop ![[SPP_J:[0-9]+]]
; CHECK: llvm.loop ![[SPP_I:[0-9]+]]
;
; for (i ...)
;   forall (i ...)
;     forall (j ...)
;
define void @spp(i64 %n0, i64 %n1, i64 %n2) {
entry:
  br label %header0

header0:
  %i0 = phi i64 [ 0, %entry ], [ %inc0, %latch0 ]
  br label %body0

body0:
  %sr1 = call token @llvm.syncregion.start()
  br label %header1

header1:
  %i1 = phi i64 [ 0, %body0 ], [ %inc1, %latch1 ]
  detach within %sr1, label %body1, label %latch1

body1:
  %sr2 = call token @llvm.syncregion.start()
  br label %header2

header2:
  %i2 = phi i64 [ 0, %body1 ], [ %inc2, %latch2 ]
  detach within %sr2, label %body2, label %latch2

body2:
  reattach within %sr2, label %latch2

latch2:
  %inc2 = add i64 %i2, 1
  %cmp2 = icmp eq i64 %inc2, %n2
  br i1 %cmp2, label %sync2, label %header2, !llvm.loop !6

sync2:
  sync within %sr2, label %reattach1

reattach1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n1
  br i1 %cmp1, label %sync1, label %header1, !llvm.loop !7

sync1:
  sync within %sr1, label %latch0

latch0:
  %inc0 = add i64 %i0, 1
  %cmp0 = icmp eq i64 %inc0, %n0
  br i1 %cmp0, label %exit, label %header0, !llvm.loop !8

exit:
  ret void
}

;-------------------------------------------------------------------------------
;
; CHECK-DAG: ![[D2:[0-9]+]] = !{!"tapir.loop.perfect.depth", i32 2}
; CHECK-DAG: ![[L1:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 1}
; CHECK-DAG: ![[L2:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 2}
; CHECK-DAG: ![[TARGET:[0-9]+]] = !{!"tapir.loop.target", i32 1}
;
;-------------------------------------------------------------------------------
;
; forall (i ...)
;   forall (j ...)
;
; CHECK-DAG: ![[PP_J]] = distinct !{![[PP_J]], ![[TARGET]], ![[L2]], {{.+}}}
; CHECK-DAG: ![[PP_I]] = distinct !{![[PP_I]], ![[TARGET]], ![[L1]], ![[D2]], {{.+}}}
;
;-------------------------------------------------------------------------------
;
; forall (i ...)
;   forall (j ...)
;     for (k ...)
;
; CHECK-DAG: ![[PPS_K]] = distinct !{![[PPS_K]]}
; CHECK-DAG: ![[PPS_J]] = distinct !{![[PPS_J]], ![[TARGET]], ![[L2]], {{.+}}}
; CHECK-DAG: ![[PPS_I]] = distinct !{![[PPS_I]], ![[TARGET]], ![[L1]], ![[D2]], {{.+}}}
;
;-------------------------------------------------------------------------------
;
; for (i ...)
;   forall (j ...)
;     forall (k ...)
;
; CHECK-DAG: ![[SPP_K]] = distinct !{![[SPP_K]], ![[TARGET]], ![[L2]], {{.+}}}
; CHECK-DAG: ![[SPP_J]] = distinct !{![[SPP_J]], ![[TARGET]], ![[L1]], ![[D2]], {{.+}}}
; CHECK-DAG: ![[SPP_I]] = distinct !{![[SPP_I]]}
;
;-------------------------------------------------------------------------------

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 1}
!2 = distinct !{!2, !1}
!3 = distinct !{!3}
!4 = distinct !{!4, !1}
!5 = distinct !{!5, !1}
!6 = distinct !{!6, !1}
!7 = distinct !{!7, !1}
!8 = distinct !{!8}

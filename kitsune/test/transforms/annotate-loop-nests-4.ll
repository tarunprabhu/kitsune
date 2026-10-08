; Check that the tapir loop annotator pass annotates loops correctly. Every
; function here contains a single loop nest. Each nest will contain exactly
; four tapir loops. The loop nest may contain other, non-tapir loops.
;
; RUN: opt -passes="kit-annotate-early" -S %s | FileCheck %s

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

; CHECK-LABEL: @spppp
; CHECK: llvm.loop ![[SPPPP_L:[0-9]+]]
; CHECK: llvm.loop ![[SPPPP_K:[0-9]+]]
; CHECK: llvm.loop ![[SPPPP_J:[0-9]+]]
; CHECK: llvm.loop ![[SPPPP_I:[0-9]+]]
; CHECK: llvm.loop ![[SPPPP_S:[0-9]+]]
;
; for (s ...)
;   forall (i ...)
;     forall (j ...)
;       forall (k ...)
;         forall (l ...)
;
define void @spppp(i64 %n0, i64 %n1, i64 %n2, i64 %n3, i64 %n4) {
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
  %sr3 = call token @llvm.syncregion.start()
  br label %header3

header3:
  %i3 = phi i64 [ 0, %body2 ], [ %inc3, %latch3 ]
  detach within %sr3, label %body3, label %latch3

body3:
  %sr4 = call token @llvm.syncregion.start()
  br label %header4

header4:
  %i4 = phi i64 [ 0, %body3 ], [ %inc4, %latch4 ]
  detach within %sr4, label %body4, label %latch4

body4:
  reattach within %sr4, label %latch4

latch4:
  %inc4 = add i64 %i4, 1
  %cmp4 = icmp eq i64 %inc4, %n4
  br i1 %cmp4, label %sync4, label %header4, !llvm.loop !5

sync4:
  sync within %sr4, label %reattach3

reattach3:
  reattach within %sr3, label %latch3

latch3:
  %inc3 = add i64 %i3, 1
  %cmp3 = icmp eq i64 %inc3, %n3
  br i1 %cmp3, label %sync3, label %header3, !llvm.loop !6

sync3:
  sync within %sr3, label %reattach2

reattach2:
  reattach within %sr2, label %latch2

latch2:
  %inc2 = add i64 %i2, 1
  %cmp2 = icmp eq i64 %inc2, %n2
  br i1 %cmp2, label %sync2, label %header2, !llvm.loop !7

sync2:
  sync within %sr2, label %reattach1

reattach1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n1
  br i1 %cmp1, label %sync1, label %header1, !llvm.loop !8

sync1:
  sync within %sr1, label %latch0

latch0:
  %inc0 = add i64 %i0, 1
  %cmp0 = icmp eq i64 %inc0, %n0
  br i1 %cmp0, label %exit, label %header0, !llvm.loop !9

exit:
  ret void
}

;-------------------------------------------------------------------------------
;
; CHECK-DAG: ![[D4:[0-9]+]] = !{!"tapir.loop.perfect.depth", i32 4}
; CHECK-DAG: ![[L1:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 1}
; CHECK-DAG: ![[L2:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 2}
; CHECK-DAG: ![[L3:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 3}
; CHECK-DAG: ![[L4:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 4}
; CHECK-DAG: ![[TARGET:[0-9]+]] = !{!"tapir.loop.target", i32 1}
;
;-------------------------------------------------------------------------------
;
; forall (i ...)
;   forall (j ...)
;     forall (k ...)
;       forall (l ...)
;
; CHECK-DAG: ![[PPPP_L]] = distinct !{![[PPPP_L]], ![[TARGET]], ![[L4]], {{.+}}}
; CHECK-DAG: ![[PPPP_K]] = distinct !{![[PPPP_K]], ![[TARGET]], ![[L3]], {{.+}}}
; CHECK-DAG: ![[PPPP_J]] = distinct !{![[PPPP_J]], ![[TARGET]], ![[L2]], {{.+}}}
; CHECK-DAG: ![[PPPP_I]] = distinct !{![[PPPP_I]], ![[TARGET]], ![[L1]], ![[D4]], {{.+}}}
;
;-------------------------------------------------------------------------------
;
; for (s ...)
;   forall (i ...)
;     forall (j ...)
;       forall (k ...)
;         forall (l ...)
;
; CHECK-DAG: ![[SPPPP_L]] = distinct !{![[SPPPP_L]], ![[TARGET]], ![[L4]], {{.+}}}
; CHECK-DAG: ![[SPPPP_K]] = distinct !{![[SPPPP_K]], ![[TARGET]], ![[L3]], {{.+}}}
; CHECK-DAG: ![[SPPPP_J]] = distinct !{![[SPPPP_J]], ![[TARGET]], ![[L2]], {{.+}}}
; CHECK-DAG: ![[SPPPP_I]] = distinct !{![[SPPPP_I]], ![[TARGET]], ![[L1]], ![[D4]], {{.+}}}
; CHECK-DAG: ![[SPPPP_S]] = distinct !{![[SPPPP_S]]}
;
;-------------------------------------------------------------------------------

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 1}
!2 = distinct !{!2, !1}
!3 = distinct !{!3, !1}
!4 = distinct !{!4, !1}
!5 = distinct !{!5, !1}
!6 = distinct !{!6, !1}
!7 = distinct !{!7, !1}
!8 = distinct !{!8, !1}
!9 = distinct !{!9}

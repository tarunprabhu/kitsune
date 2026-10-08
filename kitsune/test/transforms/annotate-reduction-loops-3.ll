; Check that the tapir loop annotator pass annotates loops correctly. Every
; function here contains a single loop nest. Each nest will contain exactly
; three tapir loops. The loop nest may contain other, non-tapir loops.
;
; RUN: opt -passes="kit-annotate-early" -S %s | FileCheck %s

; CHECK-LABEL: @ppp
; CHECK: llvm.loop ![[PPP3:[0-9]+]]
; CHECK: llvm.loop ![[PPP2:[0-9]+]]
; CHECK: llvm.loop ![[PPP1:[0-9]+]]
;
; forall (i ...)
;   forall (j ...)
;     forall (k ...)
;
define void @ppp(i64 %n) {
entry:
  %result = alloca i64
  %sr1 = call token @llvm.syncregion.start()
  br label %header1

header1:
  %i1 = phi i64 [ 0, %entry ], [ %inc1, %latch1 ]
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
  call void (i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1, i32 1, ptr %result, i64 %i3, i32 0, i32 8, i64 0, ptr null)
  br label %reattach3

reattach3:
  reattach within %sr3, label %latch3

latch3:
  %inc3 = add i64 %i3, 1
  %cmp3 = icmp eq i64 %inc3, %n
  br i1 %cmp3, label %sync3, label %header3, !llvm.loop !1

sync3:
  sync within %sr3, label %reattach2

reattach2:
  reattach within %sr2, label %latch2

latch2:
  %inc2 = add i64 %i2, 1
  %cmp2 = icmp eq i64 %inc2, %n
  br i1 %cmp2, label %sync2, label %header2, !llvm.loop !2

sync2:
  sync within %sr2, label %reattach1

reattach1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n
  br i1 %cmp1, label %sync1, label %header1, !llvm.loop !3

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

; CHECK-LABEL: @ppps
; CHECK: llvm.loop ![[PPPS0:[0-9]+]]
; CHECK: llvm.loop ![[PPPS3:[0-9]+]]
; CHECK: llvm.loop ![[PPPS2:[0-9]+]]
; CHECK: llvm.loop ![[PPPS1:[0-9]+]]
;
; forall (i ...)
;   forall (j ...)
;     forall (k ...)
;       for (l ...)
;
define void @ppps(i64 %n) {
entry:
  %result = alloca i64
  %sr1 = call token @llvm.syncregion.start()
  br label %header1

header1:
  %i1 = phi i64 [ 0, %entry ], [ %inc1, %latch1 ]
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
  br label %header0

header0:
  %i0 = phi i64 [ 0, %body3 ], [ %inc0, %header0 ]
  call void (i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1, i32 1, ptr %result, i64 %i0, i32 0, i32 8, i64 0, ptr null)
  %inc0 = add i64 %i0, 1
  %cmp0 = icmp eq i64 %inc0, %n
  br i1 %cmp0, label %reattach3, label %header0, !llvm.loop !4

reattach3:
  reattach within %sr3, label %latch3

latch3:
  %inc3 = add i64 %i3, 1
  %cmp3 = icmp eq i64 %inc3, %n
  br i1 %cmp3, label %sync3, label %header3, !llvm.loop !5

sync3:
  sync within %sr3, label %reattach2

reattach2:
  reattach within %sr2, label %latch2

latch2:
  %inc2 = add i64 %i2, 1
  %cmp2 = icmp eq i64 %inc2, %n
  br i1 %cmp2, label %sync2, label %header2, !llvm.loop !6

sync2:
  sync within %sr2, label %reattach1

reattach1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n
  br i1 %cmp1, label %sync1, label %header1, !llvm.loop !7

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

; CHECK-LABEL: @sppp
; CHECK: llvm.loop ![[SPPP3:[0-9]+]]
; CHECK: llvm.loop ![[SPPP2:[0-9]+]]
; CHECK: llvm.loop ![[SPPP1:[0-9]+]]
; CHECK: llvm.loop ![[SPPP0:[0-9]+]]
;
; for (i ...)
;   forall (j ...)
;     forall (k ...)
;       forall (l ...)
;
define void @sppp(i64 %n) {
entry:
  %result = alloca i64
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
  call void (i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1, i32 1, ptr %result, i64 %i0, i32 0, i32 8, i64 0, ptr null)
  br label %reattach3

reattach3:
  reattach within %sr3, label %latch3

latch3:
  %inc3 = add i64 %i3, 1
  %cmp3 = icmp eq i64 %inc3, %n
  br i1 %cmp3, label %sync3, label %header3, !llvm.loop !8

sync3:
  sync within %sr3, label %reattach2

reattach2:
  reattach within %sr2, label %latch2

latch2:
  %inc2 = add i64 %i2, 1
  %cmp2 = icmp eq i64 %inc2, %n
  br i1 %cmp2, label %sync2, label %header2, !llvm.loop !9

sync2:
  sync within %sr2, label %reattach1

reattach1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n
  br i1 %cmp1, label %sync1, label %header1, !llvm.loop !10

sync1:
  sync within %sr1, label %latch0

latch0:
  %inc0 = add i64 %i0, 1
  %cmp0 = icmp eq i64 %inc0, %n
  br i1 %cmp0, label %exit, label %header0, !llvm.loop !11

exit:
  ret void
}

;-------------------------------------------------------------------------------
;
; CHECK-DAG: ![[D3:[0-9]+]] = !{!"tapir.loop.perfect.depth", i32 3}
; CHECK-DAG: ![[L1:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 1}
; CHECK-DAG: ![[L2:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 2}
; CHECK-DAG: ![[L3:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 3}
; CHECK-DAG: ![[REDUCTION:[0-9]+]] = !{!"tapir.loop.reduction"}
; CHECK-DAG: ![[TARGET:[0-9]+]] = !{!"tapir.loop.target", i32 1}
;
;-------------------------------------------------------------------------------
;
; forall (i ...)
;   forall (j ...)
;     forall (k ...)
;
; CHECK-DAG: ![[PPP3]] = distinct !{![[PPP3]], ![[TARGET]], ![[L3]], {{.+}}}
; CHECK-DAG: ![[PPP2]] = distinct !{![[PPP2]], ![[TARGET]], ![[L2]], {{.+}}}
; CHECK-DAG: ![[PPP1]] = distinct !{![[PPP1]], ![[TARGET]], ![[REDUCTION]], ![[L1]], ![[D3]], {{.+}}}
;
;-------------------------------------------------------------------------------
;
; forall (i ...)
;   forall (j ...)
;     forall (k ...)
;       for (l ...)
;
; CHECK-DAG: ![[PPPS0]] = distinct !{![[PPPS0]]}
; CHECK-DAG: ![[PPPS3]] = distinct !{![[PPPS3]], ![[TARGET]], ![[L3]], {{.+}}}
; CHECK-DAG: ![[PPPS2]] = distinct !{![[PPPS2]], ![[TARGET]], ![[L2]], {{.+}}}
; CHECK-DAG: ![[PPPS1]] = distinct !{![[PPPS1]], ![[TARGET]], ![[REDUCTION]], ![[L1]], ![[D3]], {{.+}}}
;
;-------------------------------------------------------------------------------
;
; for (i ...)
;   forall (j ...)
;     forall (k ...)
;       forall (l ...)
;
; CHECK-DAG: ![[SPPP3]] = distinct !{![[SPPP3]], ![[TARGET]], ![[L3]], {{.+}}}
; CHECK-DAG: ![[SPPP2]] = distinct !{![[SPPP2]], ![[TARGET]], ![[L2]], {{.+}}}
; CHECK-DAG: ![[SPPP1]] = distinct !{![[SPPP1]], ![[TARGET]], ![[REDUCTION]], ![[L1]], ![[D3]], {{.+}}}
; CHECK-DAG: ![[SPPP0]] = distinct !{![[SPPP0]]}
;
;-------------------------------------------------------------------------------

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}
!2 = distinct !{!2, !0}
!3 = distinct !{!3, !0}
!4 = distinct !{!4}
!5 = distinct !{!5, !0}
!6 = distinct !{!6, !0}
!7 = distinct !{!7, !0}
!8 = distinct !{!8, !0}
!9 = distinct !{!9, !0}
!10 = distinct !{!10, !0}
!11 = distinct !{!11}

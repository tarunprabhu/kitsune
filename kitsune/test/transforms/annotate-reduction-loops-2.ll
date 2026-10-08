; Check that tapir reduction loops are annotated correctly. Only the outermost
; loop in a nest of tapir loops is annotated with the tapir.loop.reduction
; attribute.
;
; RUN: opt -passes="kit-annotate-early" -S %s | FileCheck %s

; forall (...)
;   forall (...)
;     kit.reduce
;
; CHECK-LABEL: @pp
; CHECK: !llvm.loop ![[PP2:[0-9]+]]
; CHECK: !llvm.loop ![[PP1:[0-9]+]]
;
define void @pp(i64 %n) {
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
  call void (i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1, i32 1, ptr %result, i64 %i2, i32 0, i32 8, i64 0, ptr null)
  reattach within %sr2, label %latch2

latch2:
  %inc2 = add i64 %i2, 1
  %cmp2 = icmp eq i64 %inc2, %n
  br i1 %cmp2, label %sync2, label %header2, !llvm.loop !1

sync2:
  sync within %sr2, label %reattach1

reattach1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n
  br i1 %cmp1, label %sync1, label %header1, !llvm.loop !2

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

; forall (...)
;   forall (...)
;     for (...)
;       kit.reduce
;
; CHECK-LABEL: @pps
; CHECK: !llvm.loop ![[PPS0:[0-9]+]]
; CHECK: !llvm.loop ![[PPS2:[0-9]+]]
; CHECK: !llvm.loop ![[PPS1:[0-9]+]]
;
define void @pps(i64 %n) {
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
  br label %header0

header0:
  %i0 = phi i64 [ 0, %body2 ], [ %inc0, %header0 ]
  call void (i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1, i32 1, ptr %result, i64 %i0, i32 0, i32 8, i64 0, ptr null)
  %inc0 = add i64 %i0, 1
  %cmp0 = icmp eq i64 %inc0, %n
  br i1 %cmp0, label %reattach2, label %header0, !llvm.loop !3

reattach2:
  reattach within %sr2, label %latch2

latch2:
  %inc2 = add i64 %i2, 1
  %cmp2 = icmp eq i64 %inc2, %n
  br i1 %cmp2, label %sync2, label %header2, !llvm.loop !4

sync2:
  sync within %sr2, label %reattach1

reattach1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n
  br i1 %cmp1, label %sync1, label %header1, !llvm.loop !5

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

; for (...)
;   forall (...)
;     forall (...)
;       kit.reduce
;
; CHECK-LABEL: @spp
; CHECK: !llvm.loop ![[SPP2:[0-9]+]]
; CHECK: !llvm.loop ![[SPP1:[0-9]+]]
; CHECK: !llvm.loop ![[SPP0:[0-9]+]]
;
define void @spp(i64 %n) {
entry:
  %result = alloca i64
  br label %header0

header0:
  %i0 = phi i64 [ 0, %entry ], [ %inc1, %latch0 ]
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
  call void (i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1, i32 1, ptr %result, i64 %i0, i32 0, i32 8, i64 0, ptr null)
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
  sync within %sr1, label %latch0

latch0:
  %inc0 = add i64 %i0, 1
  %cmp0 = icmp eq i64 %inc0, %n
  br i1 %cmp0, label %exit, label %header0, !llvm.loop !8

exit:
  ret void
}

; ------------------------------------------------------------------------------
; CHECK-DAG: ![[D2:[0-9]+]] = !{!"tapir.loop.perfect.depth", i32 2}
; CHECK-DAG: ![[L1:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 1}
; CHECK-DAG: ![[L2:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 2}
; CHECK-DAG: ![[REDUCTION:[0-9]+]] = !{!"tapir.loop.reduction"}
; CHECK-DAG: ![[TARGET:[0-9]+]] = !{!"tapir.loop.target", i32 1}
;
; ------------------------------------------------------------------------------
;
; forall (...)
;   forall (...)
;
; CHECK-DAG: ![[PP1]] = distinct !{![[PP1]], ![[TARGET]], ![[REDUCTION]], ![[L1]], {{.+}}}
; CHECK-DAG: ![[PP2]] = distinct !{![[PP2]], ![[TARGET]], ![[L2]], {{.+}}}
;
; ------------------------------------------------------------------------------
;
; forall (...)
;   forall (...)
;     for (...)
;
; CHECK-DAG: ![[PPS1]] = distinct !{![[PPS1]], ![[TARGET]], ![[REDUCTION]], ![[L1]], {{.+}}}
; CHECK-DAG: ![[PPS2]] = distinct !{![[PPS2]], ![[TARGET]], ![[L2]], {{.+}}}
; CHECK-DAG: ![[PPS0]] = distinct !{![[PPS0]]}
;
; ------------------------------------------------------------------------------
;
; for (...)
;   forall (...)
;     forall (...)
;
; CHECK-DAG: ![[SPP0]] = distinct !{![[SPP0]]}
; CHECK-DAG: ![[SPP1]] = distinct !{![[SPP1]], ![[TARGET]], ![[REDUCTION]], ![[L1]], {{.+}}}
; CHECK-DAG: ![[SPP2]] = distinct !{![[SPP2]], ![[TARGET]], ![[L2]], {{.+}}}
;
; ------------------------------------------------------------------------------

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}
!2 = distinct !{!2, !0}
!3 = distinct !{!3}
!4 = distinct !{!4, !0}
!5 = distinct !{!5, !0}
!6 = distinct !{!6, !0}
!7 = distinct !{!7, !0}
!8 = distinct !{!8}

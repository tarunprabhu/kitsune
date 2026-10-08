; Check that the kit-annotate-early pass correctly adds the tapir.loop.reduction
; attribute to tapir loops of depth 1.
;
; RUN: opt -passes="kit-annotate-early" -S %s \
; RUN:     | FileCheck %s

; CHECK-LABEL: @ps
; CHECK: !llvm.loop ![[PS0:[0-9]+]]
; CHECK: !llvm.loop ![[PS1:[0-9]+]]
;
; forall (...)
;   for (...)
define void @ps(i64 %n) {
entry:
  %result = alloca i64
  %sr1 = call token @llvm.syncregion.start()
  br label %header1

header1:
  %i1 = phi i64 [ 0, %entry ], [ %inc1, %latch1 ]
  detach within %sr1, label %body1, label %latch1

body1:
  br label %header0

header0:
  %i0 = phi i64 [ 0, %body1 ], [ %inc0, %header0 ]
  call void (i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1, i32 1, ptr %result, i64 %i0, i32 0, i32 8, i64 0, ptr null)
  %inc0 = add i64 %i0, 1
  %cmp0 = icmp eq i64 %inc0, %n
  br i1 %cmp0, label %reattach1, label %header0, !llvm.loop !1

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

; CHECK-LABEL: @sp
; CHECK-DAG: !llvm.loop ![[SP1:[0-9]+]]
; CHECK-DAG: !llvm.loop ![[SP0:[0-9]+]]
;
; for (...)
;   forall (...)
;
define void @sp(i64 %n) {
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
  call void (i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1, i32 1, ptr %result, i64 %i0, i32 0, i32 8, i64 0, ptr null)
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n
  br i1 %cmp1, label %sync1, label %header1, !llvm.loop !3

sync1:
  sync within %sr1, label %latch0

latch0:
  %inc0 = add i64 %i0, 1
  %cmp0 = icmp eq i64 %inc0, %n
  br i1 %cmp0, label %exit, label %header0, !llvm.loop !4

exit:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1}
!2 = distinct !{!2, !0}
!3 = distinct !{!3, !0}
!4 = distinct !{!4}

; CHECK-DAG: ![[D1:[0-9]+]] = !{!"tapir.loop.perfect.depth", i32 1}
; CHECK-DAG: ![[L1:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 1}
; CHECK-DAG: ![[REDUCTION:[0-9]+]] = !{!"tapir.loop.reduction"}
; CHECK-DAG: ![[TARGET:[0-9]+]] = !{!"tapir.loop.target", i32 1}

; CHECK-DAG: ![[PS0]] = distinct !{![[PS0]]}
; CHECK-DAG: ![[PS1]] = distinct !{![[PS1]], ![[TARGET]], ![[REDUCTION]], ![[L1]], {{.+}}}

; CHECK-DAG: ![[SP1]] = distinct !{![[SP1]], ![[TARGET]], ![[REDUCTION]], ![[L1]], {{.+}}}
; CHECK-DAG: ![[SP0]] = distinct !{![[SP0]]}

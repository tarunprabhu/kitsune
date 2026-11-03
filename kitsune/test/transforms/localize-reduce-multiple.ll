; Check that reduction loops with multiple reductions are localized correctly.
;
; RUN: opt -passes='kit-reduce-localize' -S %s | FileCheck %s

; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[FSUM:[^,]+]]
; CHECK-SAME: ptr %[[ISUM:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK: %[[SR1:.+]] = call token @llvm.syncregion.start
; CHECK: %[[I1:.+]] = phi i64
; CHECK: detach within %[[SR1]], label %[[BODY1:.+]], label %[[LATCH1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY1]]:
; CHECK-NEXT: %[[LOCALF:.+]] = alloca double
; CHECK-NEXT: %[[LOCALI:.+]] = alloca i64
; CHECK-NEXT: store i64 0, ptr %[[LOCALI]]
; CHECK-NEXT: store double 0.000000e+00, ptr %[[LOCALF]]
; CHECK-NEXT: br label %[[HEADER2:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER2]]:
; CHECK-NEXT: %[[I2:.+]] = phi i64
; CHECK-SAME: [ 0, %[[BODY1]] ]
; CHECK-SAME: [ %[[INC2:.+]], %[[HEADER2]] ]
; CHECK-NEXT: %[[F2:.+]] = uitofp i64 %[[I2]] to double
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 1024
; CHECK-SAME: i32 5
; CHECK-SAME: ptr %[[LOCALF]]
; CHECK-SAME: double %[[F2]]
; CHECK-SAME: i32 1
; CHECK-SAME: i32 8
; CHECK-SAME: double 0.000000e+00
; CHECK-SAME: ptr @fadd
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 1024
; CHECK-SAME: i32 5
; CHECK-SAME: ptr %[[LOCALI]]
; CHECK-SAME: i64 %[[I2]]
; CHECK-SAME: i32 1
; CHECK-SAME: i32 8
; CHECK-SAME: i64 0
; CHECK-SAME: ptr @iadd
; CHECK-NEXT: %[[INC2]] = add i64 %[[I2]], 1
; CHECK-NEXT: %[[CMP2:.+]] = icmp eq i64 %[[INC2]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP2]], label %[[REATTACH1:.+]], label %[[HEADER2]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH1]]:
; CHECK-NEXT: %[[VF:.+]] = load double, ptr %[[LOCALF]]
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 1024
; CHECK-SAME: i32 5
; CHECK-SAME: ptr %[[FSUM]]
; CHECK-SAME: double %[[VF]]
; CHECK-SAME: i32 2
; CHECK-SAME: i32 8
; CHECK-SAME: double 0.000000e+00
; CHECK-SAME: ptr @fadd
; CHECK-NEXT: %[[VI:.+]] = load i64, ptr %[[LOCALI]]
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 1024
; CHECK-SAME: i32 5
; CHECK-SAME: ptr %[[ISUM]]
; CHECK-SAME: i64 %[[VI]]
; CHECK-SAME: i32 2
; CHECK-SAME: i32 8
; CHECK-SAME: i64 0
; CHECK-SAME: ptr @iadd
define void @f(ptr %fsum, ptr %isum, i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  br label %header1

header1:
  %i1 = phi i64 [ 0, %entry ], [ %inc1, %latch1 ]
  detach within %sr, label %body1, label %latch1

body1:
  br label %header2

header2:
  %i2 = phi i64 [ 0, %body1 ], [ %inc2, %header2 ]
  %f2 = uitofp i64 %i2 to double
  call void(i32, i32, ptr, double, i32, i32, double, ptr, ...) @llvm.kit.reduce(i32 1024, i32 5, ptr %fsum, double %f2, i32 0, i32 8, double 0.0, ptr @fadd)
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1024, i32 5, ptr %isum, i64 %i2, i32 0, i32 8, i64 0, ptr @iadd)
  %inc2 = add i64 %i2, 1
  %cmp2 = icmp eq i64 %inc2, %n
  br i1 %cmp2, label %reattach1, label %header2, !llvm.loop !6

reattach1:
  reattach within %sr, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n
  br i1 %cmp1, label %sync1, label %header1, !llvm.loop !5

sync1:
  sync within %sr, label %exit

exit:
  ret void
}

declare void @fadd(ptr, double)
declare void @iadd(ptr, i64)

!0 = !{!"tapir.loop.target", i32 512}
!1 = !{!"tapir.loop.perfect.level", i32 1}
!2 = !{!"tapir.loop.perfect.depth", i32 1}
!3 = !{!"tapir.loop.reduction"}
!4 = !{!"tapir.loop.prepared"}
!5 = distinct !{!5, !0, !1, !2, !3, !4}
!6 = distinct !{!6}

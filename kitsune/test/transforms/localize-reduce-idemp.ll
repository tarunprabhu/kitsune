; If the tapir.loop.localized attribute is present on a loop, it will not be
; localized, even if it has not already been localized.
;
; RUN: opt -passes='kit-reduce-localize' -S %s | FileCheck %s

; CHECK-LABEL: @f
; CHECK-SAME: %[[RES:[^,]+]]
; CHECK-SAME: %[[N:[^)]+]]
; CHECK: %[[SR:.+]] = call token @llvm.syncregion.start()
; CHECK: detach within %[[SR]], label %[[BODY1:.+]], label
; CHECK-EMPTY:
; CHECK-NEXT: [[PH2:.+]]
; CHECK-NEXT: br label %[[HEADER2:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER2]]:
; CHECK-NEXT: %[[I2:.+]] = phi i64
; CHECK-SAME: [ 0, %{{[^ ]+}} ]
; CHECK-SAME: [ %[[INC2:.+]], %[[HEADER2]] ]
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 4
; CHECK-SAME: i32 5
; CHECK-SAME: ptr %[[RES]]
; CHECK-SAME: i64 %[[I2]]
; CHECK-SAME: i32 0
; CHECK-SAME: i32 8
; CHECK-SAME: i64 0
; CHECK-NEXT: %[[INC2]] = add i64 %[[I2]], 1
; CHECK-NEXT: %[[CMP2:.+]] = icmp eq i64 %[[INC2]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP2]], label %[[REATTACH1:.+]], label %[[HEADER2]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH1]]:
; CHECK-NEXT: reattach within %[[SR]]

define void @f(ptr %res, i64 %n) {
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
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 4, i32 5, ptr %res, i64 %i2, i32 0, i32 8, i64 0, ptr @sum)
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

declare void @sum(ptr, i64)

!0 = !{!"tapir.loop.target", i32 4}
!1 = !{!"tapir.loop.perfect.level", i32 1}
!2 = !{!"tapir.loop.perfect.depth", i32 1}
!3 = !{!"tapir.loop.reduction"}
!4 = !{!"tapir.loop.localized"}
!5 = distinct !{!5, !0, !1, !2, !3, !4}
!6 = distinct !{!6}


; The normalize pass should have no effect on non-tapir loops.
;
; RUN: opt -passes=kit-loop-normalize -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: %[[N:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: br label %[[LOOP:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOP:.+]]:
; CHECK-NEXT: %[[I:.+]] = phi i64
; CHECK-SAME: [ 0, %[[ENTRY]] ]
; CHECK-SAME: [ %[[INC:.+]], %[[LOOP]] ]
; CHECK-NEXT: %[[INC:.+]] = add i64 %[[I]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[INC]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[LOOP]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void

define void @f(i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0}

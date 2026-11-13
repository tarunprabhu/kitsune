; If a non-tapir loop is not guarded, kit-loop-guard will not add a guard.
;
; RUN: opt -passes=kit-loop-guard -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: phi i64
; CHECK-SAME: [ 0, %[[ENTRY]] ]
; CHECK-SAME: [ %{{.+}}, %[[HEADER]] ]

define void @f(i64 %n) {
entry:
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %header ]
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %exit, label %header, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0}

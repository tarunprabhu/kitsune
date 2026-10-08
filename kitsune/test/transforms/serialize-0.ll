; The kit-serialize pass should have no effect on non-tapir loops.
;
; RUN: opt -passes=kit-serialize -S %s | FileCheck %s
;
; CHECK-LABEL %f
; CHECK: [[ENTRY:.+]]
; CHECK: br label %[[HEADER:.+]]
; CHECK: [[HEADER]]:
; CHECK: phi i64
; CHECK: add
; CHECK: icmp
; CHECK: br {{.+}}, label %[[EXIT:.+]], label %[[HEADER]]
; CHECK-SAME: !llvm.loop ![[LOOP:[0-9]+]]
; CHECK: [[EXIT]]:
; CHECK: ret void
;
; CHECK: ![[LOOP]] = distinct !{![[LOOP]]}

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

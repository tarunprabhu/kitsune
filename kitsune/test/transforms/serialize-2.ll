; By default, tapir loops of depth 2 should not be serialized. We set the tapir
; target to 'serial' because tapir loops with this tapir target are not treated
; any differently from loops with a parallel tapir target (CPU, or GPU-centric).
;
; RUN: opt -passes=kit-serialize -S %s | FileCheck %s

; CHECK-LABEL: @f
; CHECK: %[[SR1:.+]] = call token @llvm.syncregion.start
; CHECK: detach within %[[SR1]]
; CHECK: %[[SR2:.+]] = call token @llvm.syncregion.start
; CHECK: detach within %[[SR2]]
; CHECK: reattach within %[[SR2]]
; CHECK: reattach within %[[SR1]]

define void @f(i64 %n) {
entry:
  %sr1 = call token @llvm.syncregion.start()
  br label %header1

header1:
  %i = phi i64 [ 0, %entry ], [ %inc1, %latch1 ]
  detach within %sr1, label %body1, label %latch1

body1:
  %sr2 = call token @llvm.syncregion.start()
  br label %header2

header2:
  %j = phi i64 [ 0, %body1 ], [ %inc2, %latch2 ]
  detach within %sr2, label %body2, label %latch2

body2:
  reattach within %sr2, label %latch2

latch2:
  %inc2 = add i64 %j, 1
  %cmp2 = icmp eq i64 %inc2, %n
  br i1 %cmp2, label %sync2, label %header2, !llvm.loop !4

sync2:
  sync within %sr2, label %reattach1

reattach1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i, 1
  %cmp1 = icmp eq i64 %inc1, %n
  br i1 %cmp1, label %sync1, label %header1, !llvm.loop !5

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1}
!1 = !{!"tapir.loop.perfect.depth", i32 2}
!2 = !{!"tapir.loop.perfect.level", i32 1}
!3 = !{!"tapir.loop.perfect.level", i32 2}
!4 = distinct !{!4, !0, !3}
!5 = distinct !{!5, !0, !1, !2}

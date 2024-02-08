; Check that a secondary induction variable that has the same type as the
; primary induction variable and is an add-recurrence is handled correctly.
;
; This pass will only compute the secondary IV from the primary and replace all
; uses of the secondary, but will not remove any code.
;
; RUN: not opt -passes='kit-ive' %s -S 2>&1 | FileCheck %s
;
; CHECK: secondary induction variable has unsupported operator 'xor'

define void @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch ]
  %j = phi i64 [ 99, %entry ], [ %next.j, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  call void @ext(i64 %j)
  reattach within %syncreg, label %latch

latch:
  %inc.i = add i64 %i, 1
  %next.j = xor i64 %j, 1
  %cmp.i = icmp eq i64 %i, %n
  br i1 %cmp.i, label %exit, label %header, !llvm.loop !1

exit:
  ret void
}

declare void @ext(i64)

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}

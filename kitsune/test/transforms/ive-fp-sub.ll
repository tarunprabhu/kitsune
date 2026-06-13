; Check that a secondary induction variable that has the same type as the
; primary induction variable and is an add-recurrence is handled correctly.
;
; This pass will only compute the secondary IV from the primary and replace all
; uses of the secondary, but will not remove any code.
;
; RUN: opt -passes='kit-ive' %s -S | FileCheck %s
;
; CHECK: %[[CIV:.+]] = phi i64
; CHECK-NEXT: %[[SECIV:.+]] = phi double
; CHECK-NEXT: %[[CSTIV:.+]] = sitofp i64 %[[CIV]] to double
; CHECK-NEXT: %[[STRIDE:.+]] = fmul double %[[CSTIV]], 5.000000e-01
; CHECK-NEXT: %[[REPL:.+]] = fsub double 9.900000e+01, %[[STRIDE]]
; CHECK-NEXT: detach within
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY:.+]]
; CHECK-NEXT: call void @ext(double %[[REPL]])
; CHECK-NEXT: reattach

define void @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch ]
  %j = phi double [ 99.0, %entry ], [ %next.j, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  call void @ext(double %j)
  reattach within %syncreg, label %latch

latch:
  %inc.i = add i64 %i, 1
  %next.j = fsub double %j, 0.5
  %cmp.i = icmp eq i64 %i, %n
  br i1 %cmp.i, label %exit, label %header, !llvm.loop !1

exit:
  ret void
}

declare void @ext(double)

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}

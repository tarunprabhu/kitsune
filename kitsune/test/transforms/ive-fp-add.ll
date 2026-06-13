; Check that a secondary induction variable that has the same type as the
; primary induction variable and is an add-recurrence is handled correctly.
;
; This pass will only compute the secondary IV from the primary and replace all
; uses of the secondary, but will not remove any code.
;
; RUN: opt -passes='kit-ive' %s -S | FileCheck %s
;
; CHECK: %[[IV:.+]] = phi i64
; CHECK-NEXT: %j = phi float
; CHECK-NEXT: %[[CSTIV:.+]] = sitofp i64 %[[IV]] to float
; CHECK-NEXT: %[[STRIDE:.+]] = fmul float %[[CSTIV]], 5.000000e-01
; CHECK-NEXT: %[[REPL:.+]] = fadd float 9.900000e+01, %[[STRIDE]]
; CHECK-NEXT: detach within
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY:.+]]
; CHECK-NEXT: call void @ext(float %[[REPL]])
; CHECK-NEXT: reattach

define void @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch ]
  %j = phi float [ 99.0, %entry ], [ %next.j, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  call void @ext(float %j)
  reattach within %syncreg, label %latch

latch:
  %inc.i = add i64 %i, 1
  %next.j = fadd float %j, 0.5
  %cmp.i = icmp eq i64 %i, %n
  br i1 %cmp.i, label %exit, label %header, !llvm.loop !1

exit:
  ret void
}

declare void @ext(float)

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}

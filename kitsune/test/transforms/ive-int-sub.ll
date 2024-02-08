; Check that a secondary induction variable that is an integer sub recurrence
; is handled correctly.
;
; RUN: opt -passes='kit-ive' %s -S | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-NEXT: [[PH:.+]]:
; CHECK-NEXT: %[[SYNCREG:.+]] = tail call token
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK: %[[PRIMIV:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PH]] ]
; CHECK-SAME: [ %[[NEXT_PRIMIV:.+]], %[[LATCH:.+]] ]
; CHECK: detach within %[[SYNCREG]], label %[[BODY:.+]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: %[[STRIDE:.+]] = mul i64 %[[PRIMIV]], 5
; CHECK-NEXT: %[[REPL:.+]] = sub i64 99, %[[STRIDE]]
; CHECK-NEXT: call void @ext(i64 %[[REPL]])
; CHECK-NEXT: reattach within %[[SYNCREG]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH]]:
; CHECK-NEXT: %[[NEXT_PRIMIV]] = add i64 %[[PRIMIV]], 1
; CHECK-NEXT: %[[CMP_PRIMIV:.+]] = icmp eq i64 %[[PRIMIV]]
; CHECK-NEXT: br i1 %[[CMP_PRIMIV]], label %[[EXIT:.+]], label %[[HEADER]]

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
  %next.j = sub i64 %j, 5
  %cmp.i = icmp eq i64 %i, %n
  br i1 %cmp.i, label %exit, label %header, !llvm.loop !1

exit:
  ret void
}

declare void @ext(i64)

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}

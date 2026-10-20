; The step of a secondary induction variable need not be a compile-time constant
; as long as it is loop-invariant.
;
; RUN: opt -passes='kit-ive' -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: i64 %[[N:[^,]+]]
; CHECK-SAME: i64 %[[STEP:[^)]+]]
; CHECK-NEXT: [[PH:.+]]:
; CHECK-NEXT: %[[SYNCREG:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK: %[[PRIMIV:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PH]] ]
; CHECK-SAME: [ %[[NEXT:.+]], %[[LATCH:.+]] ]
; CHECK: detach within %[[SYNCREG]], label %[[BODY:.+]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: %[[STRIDE:.+]] = mul i64 %[[PRIMIV]], %[[STEP]]
; CHECK-NEXT: %[[REPL:.+]] = add i64 7, %[[STRIDE]]
; CHECK-NEXT: call void @ext(i64 %[[REPL]])
; CHECK-NEXT: reattach within %[[SYNCREG]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH]]:
; CHECK-NEXT: %[[NEXT]] = add i64 %[[PRIMIV]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[NEXT]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[HEADER]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]
; CHECK-NEXT: ret void

define void @f(i64 %n, i64 %step) {
entry:
  %syncreg = call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch ]
  %j = phi i64 [ 7, %entry ], [ %next.j, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  call void @ext(i64 %j)
  reattach within %syncreg, label %latch

latch:
  %inc.i = add i64 %i, 1
  %next.j = add i64 %j, %step
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %exit, label %header, !llvm.loop !1

exit:
  ret void
}

declare void @ext(i64)

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}

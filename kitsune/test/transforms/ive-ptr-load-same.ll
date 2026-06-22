; Pointer inductions where the initial value and the updated values of the
; pointer are obtained by loading from the same location are supported. The
; semantics of a tapir loop guarantee that iterations of the loop may be
; executed independently of one another. This implies that the load will always
; return the same value in each iteration, and that value will be the same as
; the initial value of the induction variable.
;
; RUN: opt -passes='kit-ive' -S %s 2>&1 | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-NEXT: [[PH:.+]]:
; CHECK-NEXT: %[[SRC:.+]] = alloca ptr
; CHECK-NEXT: %[[SYNCREG:.+]] = tail call token @llvm.syncregion.start()
; CHECK-NEXT: %[[INIT:.+]] = load ptr, ptr %[[SRC]]
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: %[[PRIMIV:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PH]] ]
; CHECK-SAME: [ %[[NEXT_PRIMIV:.+]], %[[LATCH:.+]] ]
; CHECK-NEXT: detach within %[[SYNCREG]], label %[[BODY:.+]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: call void @ext(ptr %[[INIT]])
; CHECK-NEXT: reattach within %[[SYNCREG]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH]]:
; CHECK-NEXT: %[[NEXT_PRIMIV]] = add i64 %[[PRIMIV]], 1
; CHECK-NEXT: %[[N:.+]] = load i64, ptr %[[INIT]]
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[NEXT_PRIMIV]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[HEADER]]

define void @f() {
entry:
  %src = alloca ptr
  %syncreg = tail call token @llvm.syncregion.start()
  %init.j = load ptr, ptr %src
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch ]
  %j = phi ptr [ %init.j, %entry ], [ %next.j, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  call void @ext(ptr %j)
  reattach within %syncreg, label %latch

latch:
  %inc.i = add i64 %i, 1
  %next.j = load ptr, ptr %src
  %n = load i64, ptr %next.j
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %exit, label %header, !llvm.loop !1

exit:
  sync within %syncreg, label %end

end:
  ret void
}

declare void @ext(ptr)

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}

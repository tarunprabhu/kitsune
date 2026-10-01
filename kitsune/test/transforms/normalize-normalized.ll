; The normalize pass should have no effect on tapir loops that are already
; normalized.
;
; RUN: opt -passes=kit-loop-normalize -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: %[[N:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SYNCREG:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER:.+]]:
; CHECK-NEXT: %[[I:.+]] = phi i64
; CHECK-SAME: [ 0, %[[ENTRY]] ]
; CHECK-SAME: [ %[[INC:.+]], %[[LATCH:.+]] ]
; CHECK-NEXT: detach within %[[SYNCREG]], label %[[BODY:.+]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: reattach within %[[SYNCREG]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH]]:
; CHECK-NEXT: %[[INC:.+]] = add i64 %[[I]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[INC]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[SYNC:.+]], label %[[HEADER]]
; CHECK-EMPTY:
; CHECK-NEXT: [[SYNC]]:
; CHECK-NEXT: sync within %[[SYNCREG]], label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void

define void @f(i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %sync, label %header, !llvm.loop !0

sync:
  sync within %sr, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 1}

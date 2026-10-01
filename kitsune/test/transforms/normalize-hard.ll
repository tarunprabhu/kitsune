; Normalize tapir loops where several changes are required. In this case, the
; loop must be simplified, and the compare instruction must be fixed.
;
; RUN: opt -passes=kit-loop-normalize -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: i32 %[[N:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SYNCREG:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: %[[SKIP:.+]] = icmp sge i32 %[[N]], 0
; CHECK-NEXT: %[[N64:.+]] = zext i32 %[[N]] to i64
; CHECK-NEXT: br i1 %[[SKIP]], label %[[PH:.+]], label %[[SYNC:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH]]:
; CHECK-NEXT: %[[TC:.+]] = add {{.*}}i64 %[[N64]], 1
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK: [[HEADER]]:
; CHECK: %[[I:.+]] = phi i64
; CHECK: reattach within %[[SYNCREG]], label %[[LATCH:.+]]
; CHECK: [[LATCH]]:
; CHECK-NEXT: %[[INC:.+]] = add i64 %[[I]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[INC]], %[[TC]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[LOOPEXIT:.+]], label %[[HEADER]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOPEXIT]]:
; CHECK-NEXT: br label %[[SYNC]]
; CHECK-EMPTY:
; CHECK-NEXT: [[SYNC]]:
; CHECK-NEXT: sync within %[[SYNCREG]], label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void

define void @f(i32 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  %skip = icmp sge i32 %n, 0
  %n64 = zext i32 %n to i64
  br i1 %skip, label %header, label %sync

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %i, %n64
  br i1 %cmp, label %sync, label %header, !llvm.loop !0

sync:
  sync within %sr, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 1}

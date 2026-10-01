; Normalize tapir loops where the comparison instruction is not as expected.
;
; RUN: opt -passes=kit-loop-normalize -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: %[[N:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SYNCREG:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: %[[TC:.+]] = add i64 %[[N]], 1
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK: [[HEADER]]:
; CHECK: %[[I:.+]] = phi i64
; CHECK: reattach within %[[SYNCREG]], label %[[LATCH:.+]]
; CHECK: [[LATCH]]:
; CHECK-NEXT: %[[INC:.+]] = add i64 %[[I]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[INC]], %[[TC]]
; CHECK-NEXT: br i1 %[[CMP]], label {{.+}}, label %[[HEADER]]

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
  %cmp = icmp eq i64 %i, %n
  br i1 %cmp, label %sync, label %header, !llvm.loop !0

sync:
  sync within %sr, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 1}

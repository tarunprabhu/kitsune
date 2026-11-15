; Check that tasks with the serial tapir target are lowered correctly by the
; kit-task-spawning pass. sync instructions are not lowered by the pass.
;
; RUN: opt -passes='kit-task-spawning' -S %s | FileCheck %s

; CHECK-LABEL: @simple
; CHECK-NEXT: call token @llvm.syncregion.start()
; CHECK-NEXT: br label %[[BODY:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: br label %[[CONT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[CONT]]:
; CHECK-NEXT: ret void
define void @simple() {
  %sr = call token @llvm.syncregion.start()
  detach within %sr, label %body, label %cont, !kit.inst !1

body:
  reattach within %sr, label %cont, !kit.inst !2

cont:
  ret void
}

; CHECK-LABEL: @nested
; CHECK-NEXT: %[[SR1:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: br label %[[BODY1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY1]]:
; CHECK-NEXT: %[[SR2:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: br label %[[BODY2:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY2]]:
; CHECK-NEXT: br label %[[CONT2:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[CONT2]]:
; CHECK-NEXT: br label %[[CONT1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[CONT1]]:
; CHECK-NEXT: ret void
define void @nested() {
  %sr1 = call token @llvm.syncregion.start()
  detach within %sr1, label %body1, label %cont1, !kit.inst !3

body1:
  %sr2 = call token @llvm.syncregion.start()
  detach within %sr2, label %body2, label %cont2, !kit.inst !4

body2:
  reattach within %sr2, label %cont2, !kit.inst !5

cont2:
  reattach within %sr1, label %cont1, !kit.inst !6

cont1:
  ret void
}

!0 = !{!"kit.inst.target", i32 1}
!1 = distinct !{!1, !0}
!2 = distinct !{!2, !0}
!3 = distinct !{!3, !0}
!4 = distinct !{!4, !0}
!5 = distinct !{!5, !0}
!6 = distinct !{!6, !0}

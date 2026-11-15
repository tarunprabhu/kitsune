; Detach/Reattach instructions that are not in a tapir loop, and do not have a
; target attribute will not be lowered by the task-spawning pass.
;
; RUN: opt -passes='kit-task-spawning' -S %s | FileCheck %s

; CHECK-LABEL: @f
; CHECK-NEXT: %[[SR:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: detach within %[[SR]], label %[[BODY:.+]], label %[[CONT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: reattach within %[[SR]], label %[[CONT]]
; CHECK-EMPTY:
; CHECK-NEXT: [[CONT]]:
; CHECK-NEXT: ret void
define void @f() {
  %sr = call token @llvm.syncregion.start()
  detach within %sr, label %body, label %cont

body:
  reattach within %sr, label %cont

cont:
  ret void
}

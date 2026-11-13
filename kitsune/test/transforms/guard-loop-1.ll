; Check that the simplest case of an unguarded tapir loop is handled correctly.
; This is where the loop has a single exit block, and the preheader has a single
; predecessor. This also checks that the instructions in the preheader and exit
; blocks end up in the expected blocks.
;
; RUN: opt -passes='loop-mssa(kit-loop-guard)' -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SR:.+]] = call token @llvm.syncregion.start
; CHECK-NEXT: call void @instr_begin()
; CHECK-NEXT: br i1 true, label %[[PH:.+]], label %[[SYNC:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH]]:
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: phi i64
; CHECK-SAME: [ 0, %[[PH]] ]
; CHECK-SAME: [ %{{.+}}, %[[LATCH:.+]] ]
; CHECK: [[LATCH]]:
; CHECK: br i1 %{{.+}}, label %[[LOOPEXIT:.+]], label %[[HEADER]]
; CHECK-SAME: !llvm.loop
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOPEXIT]]:
; CHECK-NEXT: br label %[[SYNC]]
; CHECK: [[SYNC]]:
; CHECK-NEXT: call void @instr_end()
; CHECK-NEXT: sync within %[[SR]]

define void @f(i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  call void @instr_begin()
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
  call void @instr_end()
  sync within %sr, label %exit

exit:
  ret void
}

declare void @instr_begin()
declare void @instr_end()

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 1}

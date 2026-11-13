; Check that loops with a deadend exit block are guarded correctly.
;
; RUN: opt -passes='loop-mssa(kit-loop-guard)' -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SR:.+]] = call token @llvm.syncregion.start
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
; CHECK-NEXT: sync within

define void @f(i64 %n, i64 %b) {
entry:
  %sr = call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  %chk = icmp eq i64 %i, %b
  br i1 %chk, label %deadend, label %reattach

reattach:
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %sync, label %header, !llvm.loop !0

sync:
  sync within %sr, label %exit

deadend:
  unreachable

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 1}

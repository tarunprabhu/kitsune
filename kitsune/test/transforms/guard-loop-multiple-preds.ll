; If the unguarded loop preheader has multiple predecessor, check that
; kit-loop-guard inserts the guard correctly.
;
; RUN: opt -passes='loop-mssa(kit-loop-guard)' -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK: [[ENTRY:.+]]:
; CHECK: [[GETARG:.+]]:
; CHECK: %[[PARSED:.+]] = call i64 @atol
; CHECK: %[[N:.+]] = phi i64
; CHECK-SAME: [ 1024, %[[ENTRY]] ]
; CHECK-SAME: [ %[[PARSED]], %[[GETARG]] ]
; CHECK-NEXT: br i1 true, label %[[PH:.+]], label %[[SYNC:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH]]:
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: phi i64
; CHECK: br i1 %{{.+}}, label %[[LOOPEXIT:.+]], label %[[HEADER]]
; CHECK-SAME: !llvm.loop
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOPEXIT]]:
; CHECK-NEXT: br label %[[SYNC]]
; CHECK-EMPTY:
; CHECK-NEXT: [[SYNC]]
; CHECK-NEXT: sync within

define void @f(i32 %argc, ptr %argv) {
entry:
  %sr = call token @llvm.syncregion.start()
  %hasarg = icmp ugt i32 %argc, 0
  br i1 %hasarg, label %getarg, label %ph

getarg:
  %parsed = call i64 @atol(ptr %argv)
  br label %ph

ph:
  %n = phi i64 [ 1024, %entry ], [ %parsed, %getarg ]
  br label %header

header:
  %i = phi i64 [ 0, %ph ], [ %inc, %latch ]
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

declare i64 @atol(ptr %argv)

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 1}

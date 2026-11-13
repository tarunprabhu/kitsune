; If a tapir loop is already guarded, kit-loop-guard has no effect.
;
; RUN: opt -passes=kit-loop-guard -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: %[[N:[^)]+]]
; CHECK: %[[CMPN:.+]] = icmp eq i64 %[[N]], 0
; CHECK-NEXT: br i1 %[[CMPN]], label %[[SYNC:.+]], label %[[PH:.+]]
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
; CHECK: [[SYNC]]:
; CHECK-NEXT: sync within

define void @f(i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  %cmpn = icmp eq i64 %n, 0
  br i1 %cmpn, label %sync, label %ph

ph:
  br label %header

header:
  %i = phi i64 [ 0, %ph ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %loopexit, label %header, !llvm.loop !0

loopexit:
  br label %sync

sync:
  sync within %sr, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 1}

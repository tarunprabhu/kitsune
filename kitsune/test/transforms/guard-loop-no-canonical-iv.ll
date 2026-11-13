; If a tapir loop is already guarded, kit-loop-guard has no effect.
;
; RUN: not opt -passes=kit-loop-guard -S %s 2>&1 | FileCheck %s
;
; CHECK: tapir loop does not have a canonical induction variable

define void @f(i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 1, %entry ], [ %inc, %latch ]
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

; Tapir loops must have a finite trip count. This does not mean that the trip
; count is known at compile time, it only needs to be loop-invariant and
; computed outside the loop.
;
; RUN: not opt --tapir=nolo -passes=kit-verify-prelower %s 2>&1 | FileCheck %s

; CHECK: tapir loop trip count is not finite

; In this test, the trip count is loop invariant, but it has not been hoisted
; outside the loop. Such behavior has been seen when LICM does not hoist the
; %tc instruction for some reason. This can also happen if the DeLICM pass sinks
; %tc back into the tapir loop.
define void @p(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  reattach within %syncreg, label %latch

latch:
  %inc.i = add i64 %i, 1
  %tc = sub i64 %n, 2
  %cmp.i = icmp eq i64 %inc.i, %tc
  br i1 %cmp.i, label %sync, label %header, !llvm.loop !0

sync:
  sync within %syncreg, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 1024}

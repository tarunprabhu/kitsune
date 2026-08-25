; The only instructions allowed in the latch of a tapir loop are those that
; update the sole induction variable and check if the loop termination condition
; has been met.
;
; RUN: not opt -passes=kit-verify-prelower %s 2>&1 | FileCheck %s
;
; CHECK: tapir loop latch contains unexpected instruction: <call ext>

define void @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  reattach within %syncreg, label %latch

latch:
  call void @ext()
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %exit, label %header, !llvm.loop !1

exit:
  sync within %syncreg, label %end

end:
  ret void
}

declare void @ext()

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}

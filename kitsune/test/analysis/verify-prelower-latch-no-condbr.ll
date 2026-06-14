; The terminator of the tapir loop preheader must be an unconditional branch.
;
; RUN: not opt --tapir=nolo -passes='kit-verify-prelower' %s 2>&1 | FileCheck %s
;
; CHECK: tapir loop latch must be terminated by a conditional branch

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
  %inc.i = add i64 %i, 1
  switch i64 %inc.i, label %header [
    i64 12, label %exit
  ], !llvm.loop !1

exit:
  sync within %syncreg, label %end

end:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1024}
!1 = distinct !{!1, !0}

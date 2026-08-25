; The terminator of the tapir loop preheader must be an unconditional branch.
;
; RUN: not opt -passes='kit-verify-prelower' %s -disable-output 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: tapir loop preheader must be terminated by a unconditional branch

define void @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  indirectbr ptr null, [label %header]

header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  reattach within %syncreg, label %latch

latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %exit, label %header, !llvm.loop !1

exit:
  sync within %syncreg, label %end

end:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1024}
!1 = distinct !{!1, !0}

; The only use of the secondary IV of a tapir loop in the latch must produce
; the updated value.
;
; RUN: not opt -passes='kit-ive' -S %s 2>&1 | FileCheck %s
;
; CHECK: tapir loop induction variable not updated in latch

define void @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch ]
  %j = phi i32 [ 99, %entry ], [ %next.j, %latch ]
  detach within %syncreg, label %body, label %cont

body:
  reattach within %syncreg, label %cont

cont:
  %next.j = add i32 %j, 5
  br label %latch

latch:
  call void @ext(i32 %j)
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %i, %n
  br i1 %cmp.i, label %exit, label %header, !llvm.loop !1

exit:
  ret void
}

declare void @ext(i32)

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}

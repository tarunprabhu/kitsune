; The step of a secondary induction variable must be loop-invariant.
;
; RUN: not opt -passes='kit-ive' -S %s 2>&1 | FileCheck %s
;
; CHECK: secondary induction variable has loop-variant step

define void @f(i64 %n) {
entry:
  %syncreg = call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch ]
  %j = phi i64 [ 7, %entry ], [ %next.j, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  reattach within %syncreg, label %latch

latch:
  %step = call i64 @ext(i64 %j)
  %inc.i = add i64 %i, 1
  %next.j = add i64 %j, %step
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %exit, label %header, !llvm.loop !1

exit:
  ret void
}

declare i64 @ext(i64)

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}

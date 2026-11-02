; Check that the correct diagnostic is emitted when a tapir loop has more than
; one induction variable.
;
; RUN: not opt -passes='kit-verify-prelower' %s -disable-output 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: tapir loop must have exactly one induction variable

define void @f1(i64 %n) {
entry:
  %sr = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i1 = phi i64 [ 0, %entry ], [ %inc1, %latch ]
  %i2 = phi i64 [ 99, %entry ], [ %inc2, %latch ]
  detach within %sr, label %body, label %latch

body:
  reattach within %sr, label %latch

latch:
  %inc1 = add i64 %i1, 1
  %inc2 = add i64 %i2, 1
  %cmp = icmp eq i64 %inc1, %n
  br i1 %cmp, label %sync, label %header, !llvm.loop !3

sync:
  sync within %sr, label %exit

exit:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1024}
!1 = !{!"tapir.loop.perfect.level", i32 1}
!2 = !{!"tapir.loop.perfect.depth", i32 1}
!3 = !{!3, !0, !1, !2}

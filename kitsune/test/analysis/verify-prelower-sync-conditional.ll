; Check that the pre-lowering verifier emits the correct diagnostic when the
; sync instruction for a tapir loop does not immediately follow the loop.
;
; RUN: not opt -passes='kit-verify-prelower' %s -disable-output 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: tapir loop not post-dominated by sync instruction

define void @f1(i64 %n) {
entry:
  %sr = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %loopexit, label %header, !llvm.loop !4

loopexit:
  %cmp.n = icmp sgt i64 %n, 21
  br i1 %cmp.n, label %sync, label %exit

sync:
  sync within %sr, label %exit

exit:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1}
!1 = !{!"tapir.loop.perfect.level", i32 1}
!2 = !{!"tapir.loop.perfect.depth", i32 1}
!3 = !{!"loop.name", !"f1.loop.i"}
!4 = !{!4, !0, !1, !2, !3}

; Check that the syncregion passed to the detach, reattach and sync instructions
; are the result of a call to the llvm.syncregion.start intrinsic.
;
; RUN: not opt -passes='kit-verify-prelower' %s -disable-output 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: syncregion is not the result of an intrinsic call
;
define void @f1(i64 %n) {
entry:
  %sr = tail call token @llvm.syncregion.start()
  %sr.cst = bitcast token %sr to token
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  detach within %sr.cst, label %body, label %latch

body:
  reattach within %sr.cst, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %sync, label %header, !llvm.loop !4

sync:
  sync within %sr.cst, label %exit

exit:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1}
!1 = !{!"tapir.loop.perfect.level", i32 1}
!2 = !{!"tapir.loop.perfect.depth", i32 1}
!3 = !{!"loop.name", !"f1.loop.i"}
!4 = !{!4, !0, !1, !2, !3}

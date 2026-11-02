; All uses of the induction variable of a tapir loop must be within the loop.
;
; RUN: not opt -passes='kit-prepare' -S %s 2>&1 | FileCheck %s
;
; CHECK: tapir loop induction variable used outside tapir loop

define void @acc(i64 %n) {
entry:
  %sr = tail call token @llvm.syncregion.start()
  %cmpn1 = icmp ugt i64 0, %n
  br i1 %cmpn1, label %ph, label %sync

ph:
  br label %header

header:
  %j = phi i64 [ 0, %ph ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  reattach within %sr, label %latch

latch:
  %inc = add i64 %j, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %loopexit, label %header, !llvm.loop !0

loopexit:
  %tmp = phi i64 [ %j, %latch ]
  br label %sync

sync:
  sync within %sr, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.target", i32 1024}
!2 = !{!"tapir.loop.perfect.depth", i32 1}
!3 = !{!"tapir.loop.perfect.level", i32 1}

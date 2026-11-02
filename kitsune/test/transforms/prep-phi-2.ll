; A diagnostic must not be emitted when preparing a tapir loop with more than
; one induction variable.
;
; RUN: opt --tapir=serial -passes='kit-prepare' %s -disable-output 2>&1 \
; RUN:     | FileCheck %s --allow-empty
;
; CHECK-NOT: {{^.+$}}

define void @f(i64 %n) {
entry:
  %sr = tail call token @llvm.syncregion.start()
  %cmpn = icmp ugt i64 %n, 0
  br i1 %cmpn, label %ph, label %sync

ph:
  br label %header

header:
  %i1 = phi i64 [ 0, %ph ], [ %inc1, %latch ]
  %i2 = phi i32 [ 99, %ph ], [ %inc2, %latch ]
  detach within %sr, label %body, label %latch

body:
  reattach within %sr, label %latch

latch:
  %inc1 = add i64 %i1, 1
  %inc2 = add i32 %i2, 1
  %cmp1 = icmp eq i64 %inc1, %n
  br i1 %cmp1, label %loopexit, label %header, !llvm.loop !0

loopexit:
  br label %sync

sync:
  sync within %sr, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"tapir.loop.target", i32 1}
!2 = !{!"loop.name", !"f1.loop.i"}
!3 = !{!"tapir.loop.perfect.depth", i32 1}
!4 = !{!"tapir.loop.perfect.level", i32 1}

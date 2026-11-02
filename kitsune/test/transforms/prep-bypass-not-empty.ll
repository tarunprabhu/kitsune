; When preparing a tapir loop nest, the root must have an empty bypass block
;
; RUN: not opt -passes=kit-prepare %s 2>&1 | FileCheck %s
;
; CHECK: tapir loop bypass block is not empty

define void @f(i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  %cmpn = icmp eq i64 0, %n
  br i1 %cmpn, label %sync, label %ph

ph:
  br label %header

header:
  %i = phi i64 [ 0, %ph ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %loopexit, label %header, !llvm.loop !0

loopexit:
  br label %sync

sync:
  call void @ext()
  sync within %sr, label %exit

exit:
  ret void
}

declare void @ext()

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.target", i32 1}
!2 = !{!"tapir.loop.perfect.depth", i32 1}
!3 = !{!"tapir.loop.perfect.level", i32 1}

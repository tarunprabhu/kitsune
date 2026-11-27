; When preparing tapir reductions, it is an error if the destination of the
; reduction loop is loop-variant, and not in Kitsune's mobile address space.
;
; RUN: not opt -passes='kit-prepare' -S %s 2>&1 | FileCheck %s
;
; CHECK: error: destination of reduction, 'dest' is not loop-invariant

define void @f(ptr %parts, i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  %cmpn = icmp ne i64 %n, 0
  br i1 %cmpn, label %ph, label %sync

ph:
  br label %header

header:
  %i = phi i64 [ 0, %ph ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  %zero.or.one = and i64 %i, 1
  %dest = getelementptr i64, ptr %parts, i64 %zero.or.one
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 2, i32 5, ptr %dest, i64 %i, i32 0, i32 8, i64 0, ptr @sum)
  br label %reattach

reattach:
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %loopexit, label %header, !llvm.loop !4

loopexit:
  br label %sync

sync:
  sync within %sr, label %exit

exit:
  ret void
}

declare void @sum(ptr, i64)

!0 = !{!"tapir.loop.target", i32 2}
!1 = !{!"tapir.loop.perfect.level", i32 1}
!2 = !{!"tapir.loop.perfect.depth", i32 1}
!3 = !{!"tapir.loop.reduction"}
!4 = distinct !{!4, !0, !1, !2, !3}

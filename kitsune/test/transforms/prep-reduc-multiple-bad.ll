; The destination of a reduction cannot be used in two reductions in the same
; loop with different reduce operators.
;
; RUN: not opt -passes=kit-prepare %s 2>&1 | FileCheck %s
;
; CHECK: value 'res' used as destination of inconsistent reduce operations

declare void @sum(ptr, i64)
declare void @or(ptr, i64)

define void @f2(ptr %res, i64 %n) {
entry:
  %sr = tail call token @llvm.syncregion.start()
  %cmpn = icmp ult i64 0, %n
  br i1 %cmpn, label %ph, label %sync

ph:
  br label %header

header:
  %i = phi i64 [ 0, %ph ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1, i32 5, ptr %res, i64 %i, i32 0, i32 8, i64 0, ptr @sum)
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1, i32 2, ptr %res, i64 %i, i32 0, i32 8, i64 0, ptr @or)
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %loopexit, label %header, !llvm.loop !0

loopexit:
  br label %sync

sync:
  sync within %sr, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"tapir.loop.target", i32 1}
!2 = !{!"tapir.loop.reduction"}
!3 = !{!"tapir.loop.perfect.depth", i32 1}
!4 = !{!"tapir.loop.perfect.level", i32 1}

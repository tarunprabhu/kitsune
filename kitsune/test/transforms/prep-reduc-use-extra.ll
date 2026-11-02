; The destination of a reduction cannot be used as in the extra arguments of a
; reduction.
;
; RUN: not opt -passes=kit-prepare %s 2>&1 | FileCheck %s
;
; CHECK: value 'res' must only be used as destination of reduction in loop

declare void @custom(ptr, ptr, ptr)

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
  %iptr = inttoptr i64 %i to ptr
  call void(i32, i32, ptr, ptr, i32, i32, ptr, ptr, ...) @llvm.kit.reduce(i32 1, i32 5, ptr %res, ptr %iptr, i32 0, i32 8, ptr null, ptr @custom, ptr %res)
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

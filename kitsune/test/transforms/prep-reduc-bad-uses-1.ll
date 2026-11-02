; The destination of a reduction must only be used as the destination of a call
; instruction in a loop.
;
; RUN: not opt -passes=kit-prepare %s 2>&1 | FileCheck %s
;
; CHECK: value 'res1' must only be used as destination of reduction in loop
; CHECK: from loop 'bad-inst'

declare void @sum(ptr %res, i64 %v)

define void @f1(ptr %res1, i64 %n1) {
entry:
  %sr1 = call token @llvm.syncregion.start()
  %cmpn = icmp ult i64 0, %n1
  br i1 %cmpn, label %ph1, label %sync1

ph1:
  br label %header1

header1:
  %i1 = phi i64 [ 0, %ph1 ], [ %inc1, %latch1 ]
  detach within %sr1, label %body1, label %latch1

body1:
  %init = load i32, ptr %res1
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1, i32 5, ptr %res1, i64 %i1, i32 0, i32 8, i64 0, ptr @sum)
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n1
  br i1 %cmp1, label %loopexit1, label %header1, !llvm.loop !0

loopexit1:
  br label %sync1

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"tapir.loop.target", i32 1}
!2 = !{!"tapir.loop.reduction"}
!3 = !{!"tapir.loop.name", !"bad-inst"}
!4 = !{!"tapir.loop.perfect.depth", i32 1}
!5 = !{!"tapir.loop.perfect.level", i32 1}

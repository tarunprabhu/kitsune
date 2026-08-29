; The destination of a reduction must only be used as the destination of a call
; instruction in a loop.
;
; RUN: not opt -passes=kit-prepare %s 2>&1 | FileCheck %s
;
; CHECK: value 'res2' must only be used as destination of reduction in loop
; CHECK: from loop 'bad-call'

declare void @ext(ptr)
declare void @sum(ptr %res, i64 %v)

define void @f2(i64 %n) {
entry:
  %res2 = alloca i64
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
  call void @ext(ptr %res2)
  call void(i32, i32, ptr, i32, i64, i64, ptr, ...) @llvm.kit.reduce.0(i32 1, i32 5, ptr %res2, i32 8, i64 %i, i64 0, ptr @sum)
  reattach within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !2

for.i.exit:
  sync within %syncreg, label %for.i.end

for.i.end:
  ret void
}

!0 = !{!"tapir.loop.reduction"}
!1 = !{!"tapir.loop.target", i32 1}
!2 = distinct !{!2, !0, !1, !3}
!3 = !{!"tapir.loop.name", !"bad-call"}

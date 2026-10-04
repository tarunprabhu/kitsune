; The destination of a reduction can be used more than once in a loop as long as
; the reduction operations are the same. We don't check the output here - just
; that it is not an error.
;
; RUN: opt -passes=kit-prepare %s -disable-output 2>&1 \
; RUN:     | FileCheck %s --allow-empty
;
; CHECK-NOT: {{^.+$}}

declare void @sum(ptr, i64)

define void @f2(i64 %n) {
entry:
  %res2 = alloca i64
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1, i32 5, ptr %res2, i64 %i, i32 0, i32 8, i64 0, ptr @sum)
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1, i32 5, ptr %res2, i64 %i, i32 0, i32 8, i64 0, ptr @sum)
  reattach within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  sync within %syncreg, label %for.i.end

for.i.end:
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"tapir.loop.reduction"}
!2 = !{!"tapir.loop.target", i32 1}
!3 = distinct !{!"tapir.loop.perfect.depth", i32 1}
!4 = distinct !{!"tapir.loop.perfect.level", i32 1}

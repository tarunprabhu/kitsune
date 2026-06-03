; Check that the correct diagnostic is emitted when a tapir reduction loop has
; more than one induction variable.
;
; RUN: not opt --tapir=serial -passes='kit-reductions' %s 2>&1 \
; RUN:     -disable-output \
; RUN:     | FileCheck %s
;
; CHECK: tapir loop must have at most one induction variable

declare void @mul(ptr, i32)

define void @f1(ptr %r, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  %i.2 = phi i32 [ 99, %entry ], [ %inc.i.2, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
  call void(i32, ptr, i32, i32, i32, ptr, ...) @llvm.kit.reduce.0(i32 1, ptr %r, i32 4, i32 %i.2, i32 1, ptr @mul)
  reattach within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %inc.i.2 = add i32 %i.2, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  sync within %syncreg, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.target", i32 1}
!2 = !{!"loop.name", !"f1.loop.i"}
!3 = !{!"tapir.loop.reduction"}

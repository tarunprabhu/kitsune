; A diagnostic must not be emitted when preparing a tapir loop with more than
; one induction variable.
;
; RUN: opt --tapir=serial -passes='kit-prepare' %s -disable-output 2>&1 \
; RUN:     | FileCheck %s --allow-empty
;
; CHECK-NOT: {{^.+$}}

define void @f1(ptr %r, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  %i.2 = phi i32 [ 99, %entry ], [ %inc.i.2, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
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
!2 = distinct !{!"tapir.loop.perfect.depth", i32 1}
!3 = distinct !{!"tapir.loop.perfect.level", i32 1}

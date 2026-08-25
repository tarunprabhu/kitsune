; Multiple loops in a syncregion are allowed prior to lowering. This is because
; task-simplify will have run earlier in the pipeline and may have merged the
; syncregions.
;
; RUN: opt -passes='kit-verify-prelower' %s -disable-output 2>&1 \
; RUN:     | FileCheck %s --allow-empty
;
; CHECK-NOT: {{.+}}

define void @f1(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
  reattach within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  sync within %syncreg, label %for.i2.ph

for.i2.ph:
  br label %for.i2.header

for.i2.header:
  %i2 = phi i64 [ 0, %for.i2.ph ], [ %inc.i2, %for.i2.latch ]
  detach within %syncreg, label %for.i2.body, label %for.i2.latch

for.i2.body:
  reattach within %syncreg, label %for.i2.latch

for.i2.latch:
  %inc.i2 = add i64 %i2, 1
  %cmp.i2 = icmp eq i64 %inc.i2, %n
  br i1 %cmp.i2, label %for.i2.exit, label %for.i2.header, !llvm.loop !3

for.i2.exit:
  sync within %syncreg, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.target", i32 1024}
!2 = !{!"loop.name", !"f1.loop.i"}
!3 = distinct !{!3, !1, !4}
!4 = !{!"loop.name", !"f1.loop.i2"}

; phi nodes are fine in the body of a tapir loop. This is checking that we don't
; accidentally count the number of phi nodes in the loop body when checking that
; the tapir loop has exactly one induction variable.
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
  %cmp.n = icmp sgt i64 %n, 0
  br i1 %cmp.n, label %left, label %right

left:
  br label %for.i.reattach

right:
  br label %for.i.reattach

for.i.reattach:
  %v = phi i64 [ 0, %left ], [ 1, %right ]
  tail call void @ext1(i64 %v)
  reattach within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  sync within %syncreg, label %exit

exit:
  ret void
}

declare void @ext1(i64)

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.target", i32 1024}
!2 = !{!"loop.name", !"f1.loop.i"}

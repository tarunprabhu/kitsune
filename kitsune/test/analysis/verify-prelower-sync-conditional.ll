; Check that the pre-lowering verifier emits the correct diagnostic when the
; sync instruction for a tapir loop does not immediately follow the loop.
;
; RUN: not opt -passes='kit-verify-prelower' %s -disable-output 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: tapir loop not post-dominated by sync instruction

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
  %cmp.n = icmp sgt i64 %n, 21
  br i1 %cmp.n, label %for.i.sync, label %exit

for.i.sync:
  sync within %syncreg, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.target", i32 1024}
!2 = !{!"loop.name", !"f1.loop.i"}

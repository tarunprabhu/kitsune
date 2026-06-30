; Tapir loops must be in loop-simplify form before the kit-prepare pass is run.
; The loop here is not in simplify-form because the unique loop exit block is
; not dominated by the header.
;
; RUN: not opt -passes='kit-prepare' -S %s 2>&1 | FileCheck %s
;
; CHECK: loop is not in loop-simplify form

define void @acc(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp.n = icmp eq i64 %n, 0
  br i1 %cmp.n, label %for.j.exit, label %for.j.ph

for.j.ph:
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.j.ph ], [ %inc.j, %for.j.latch ]
  detach within %syncreg, label %for.j.body, label %for.j.latch

for.j.body:
  reattach within %syncreg, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !1

for.j.exit:
  sync within %syncreg, label %for.j.end

for.j.end:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1024}
!1 = distinct !{!1, !0}

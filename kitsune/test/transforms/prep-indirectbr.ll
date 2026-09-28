; Tapir loops must be safe to clone.
;
; RUN: not opt -passes='kit-prepare' -S %s 2>&1 | FileCheck %s
;
; CHECK: tapir loop is not safe to clone

define void @acc(i64 %n, ptr %jump) {
entry:
  %result = alloca i64
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %entry ], [ %inc.j, %for.j.latch ]
  detach within %syncreg, label %for.j.body, label %for.j.latch

for.j.body:
  indirectbr ptr %jump, [label %for.j.body1, label %for.j.body2]

for.j.body1:
  br label %for.j.reattach

for.j.body2:
  br label %for.j.reattach

for.j.reattach:
  reattach within %syncreg, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !0

for.j.exit:
  sync within %syncreg, label %for.j.end

for.j.end:
  ret void
}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.target", i32 1024}
!2 = distinct !{!"tapir.loop.perfect.depth", i32 1}
!3 = distinct !{!"tapir.loop.perfect.level", i32 1}

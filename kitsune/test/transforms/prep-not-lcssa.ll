; Tapir loops must be in LCSSA form before the kit-prepare pass is run.
;
; RUN: not opt -passes='kit-prepare' -S %s 2>&1 | FileCheck %s
;
; CHECK: loop is not in LCSSA form

declare i64 @get(i64)
declare void @put(i64)

define void @acc(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %entry ], [ %inc.j, %for.j.latch ]
  detach within %syncreg, label %for.j.body, label %for.j.latch

for.j.body:
  reattach within %syncreg, label %for.j.latch

for.j.latch:
  %s = call i64 @get(i64 %j)
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !0

for.j.exit:
  sync within %syncreg, label %for.j.end

for.j.end:
  call void @put(i64 %s)
  ret void
}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.target", i32 1024}
!2 = distinct !{!"tapir.loop.perfect.depth", i32 1}
!3 = distinct !{!"tapir.loop.perfect.level", i32 1}

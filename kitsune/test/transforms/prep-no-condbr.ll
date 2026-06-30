; Branch of tapir loop must be a conditional branch.
;
; RUN: not opt -passes='kit-prepare' -S %s 2>&1 | FileCheck %s
;
; CHECK: tapir loop latch must be terminated by a conditional branch

define void @acc(i64 %n) {
entry:
  %result = alloca i64
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %entry ], [ %inc.j, %for.j.latch ]
  detach within %syncreg, label %for.j.body, label %for.j.latch

for.j.body:
  reattach within %syncreg, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  switch i64 %inc.j, label %for.j.header [
    i64 12, label %for.j.exit
  ], !llvm.loop !1

for.j.exit:
  sync within %syncreg, label %for.j.end

for.j.end:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1024}
!1 = distinct !{!1, !0}

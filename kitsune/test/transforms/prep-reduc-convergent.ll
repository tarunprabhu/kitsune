; Tapir reduction loops may not have convergent operations.
;
; RUN: not opt -passes='kit-reductions' -S %s 2>&1 | FileCheck %s
;
; CHECK: tapir loop contains convergent operation

declare void @sum(ptr %res, i64 %v)

define void @acc(i64 %n) {
entry:
  %result = alloca i64
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %entry ], [ %inc.j, %for.j.latch ]
  detach within %syncreg, label %for.j.body, label %for.j.latch

for.j.body:
  %v = call i64 @conv(i64 %j) convergent
  call void(i32, ptr, i32, i64, i64, ptr, ...) @llvm.kit.reduce.0(i32 1024, ptr %result, i32 8, i64 %j, i64 0, ptr @sum)
  reattach within %syncreg, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, 2
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !2

for.j.exit:
  sync within %syncreg, label %for.j.end

for.j.end:
  ret void
}

declare i64 @conv(i64) convergent

!0 = !{!"tapir.loop.target", i32 1024}
!1 = !{!"tapir.loop.reduction"}
!2 = distinct !{!2, !0, !1}

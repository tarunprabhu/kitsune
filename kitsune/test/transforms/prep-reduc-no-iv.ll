; Tapir reduction loops must have a primary induction variable that is an
; integer.
;
; RUN: not opt -passes='kit-reductions' -S %s 2>&1 | FileCheck %s
;
; CHECK: primary induction variable not found in tapir loop

define void @sum(ptr %res, i64 %v) {
  %1 = load i64, ptr %res
  %2 = add i64 %1, %v
  store i64 %2, ptr %res
  ret void
}

define void @acc(double %n) {
entry:
  %result = alloca i64
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.j.header

for.j.header:
  %j = phi double [ 0.0, %entry ], [ %inc.j, %for.j.latch ]
  detach within %syncreg, label %for.j.body, label %for.j.latch

for.j.body:
  %j64 = fptosi double %j to i64
  call void(i32, ptr, i32, i64, i64, ptr, ...) @llvm.kit.reduce.0(i32 1024, ptr %result, i32 8, i64 %j64, i64 0, ptr @sum)
  reattach within %syncreg, label %for.j.latch

for.j.latch:
  %inc.j = fadd double %j, 1.0
  %cmp.j = fcmp oeq double %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !2

for.j.exit:
  sync within %syncreg, label %for.j.end

for.j.end:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1024}
!1 = !{!"tapir.loop.reduction"}
!2 = distinct !{!2, !0, !1}

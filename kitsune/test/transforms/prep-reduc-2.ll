; Check that the kit-prepare pass exits with an error if a tapir reduction
; loop is nested within another tapir loop.
;
; RUN: not opt -passes="kit-prepare" -S %s 2>&1 | FileCheck %s
;
; CHECK: tapir reduction loop must be a top-level loop
; CHECK-NEXT: from loop 'nested-tapir-loop'
; CHECK-NEXT: from function 'f'

declare void @sum(ptr %res, i64 %v)

define void @f(i64 %n) {
entry:
  %result = alloca i64
  %syncreg.i = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.j.ph, label %for.i.latch

for.j.ph:
  %syncreg.j = tail call token @llvm.syncregion.start()
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.j.ph ], [ %inc.j, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  call void (i32, i32, ptr, i32, i64, i64, ptr, ...) @llvm.kit.reduce.0(i32 1, i32 0, ptr %result, i32 8, i64 %j, i64 0, ptr @sum)
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !2

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !1

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

!0 = !{!"tapir.loop.target", i32 512}
!1 = distinct !{!1, !0}
!2 = distinct !{!2, !0, !3, !4}
!3 = !{!"tapir.loop.name", !"nested-tapir-loop"}
!4 = !{!"tapir.loop.reduction"}

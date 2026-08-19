; Reductions inside nested loops are supported, as long as the outer lopo is a
; tapir loop, and the inner loop is a serial loop.
;
; RUN: opt -passes="kit-annotate-early" -S %s | FileCheck %s

; forall (...)
;   for (...)
;     kit.reduce.0
;
; CHECK-LABEL: @ps
; CHECK: !llvm.loop ![[ACC_J:[0-9]+]]
; CHECK: !llvm.loop ![[ACC_I:[0-9]+]]
;
define void @ps(i64 %n) {
entry:
  %result = alloca i64
  %syncreg.i = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.j.ph, label %for.i.latch

for.j.ph:
  br label %for.j

for.j:
  %j = phi i64 [ 0, %for.j.ph ], [ %inc.j, %for.j ]
  call void (i32, i32, ptr, i32, i64, i64, ptr, ...) @llvm.kit.reduce.0(i32 1, i32 1, ptr %result, i32 8, i64 %j, i64 0, ptr null)
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j, !llvm.loop !2

for.j.exit:
  br label %for.j.end

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

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}
!2 = distinct !{!2}

; CHECK-DAG: ![[REDUCTION:[0-9]+]] = !{!"tapir.loop.reduction"}
; CHECK-DAG: ![[TARGET:[0-9]+]] = !{!"tapir.loop.target", i32 1}
; CHECK-DAG: ![[ACC_I]] = distinct !{![[ACC_I]], {{.*}}![[REDUCTION]]{{[,}]}}
; CHECK-DAG: ![[ACC_J]] = distinct !{![[ACC_J]]
; CHECK-NOT: ![[REDUCTION]]

; Check that the kit-annotate-early pass correctly adds the tapir.loop.reduction
; attribute to tapir loops of depth 1.
;
; RUN: opt -passes="kit-annotate-early" -S %s \
; RUN:     | FileCheck %s

declare void @sum(ptr %res, i64 %v)

; The tapir loop does not call @llvm.kit.reduce.0. It should not be annotated.
;
; CHECK-LABEL: @noacc
; CHECK: !llvm.loop ![[NOACC_J:[0-9]+]]
; CHECK: !llvm.loop ![[NOACC_I:[0-9]+]]
;
define void @noacc(i64 %n) {
entry:
  %syncreg = call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.header ], [ %inc.j, %for.j.latch ]
  detach within %syncreg, label %for.j.body, label %for.j.latch

for.j.body:
  reattach within %syncreg, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !2

for.j.exit:
  sync within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !1

for.i.exit:
  ret void
}

; The tapir loop calls @llvm.kit.reduce. It should be annotated.
;
; CHECK-LABEL: @acc
; CHECK: !llvm.loop ![[ACC_J:[0-9]+]]
; CHECK: !llvm.loop ![[ACC_I:[0-9]+]]
;
define void @acc(i64 %n) {
entry:
  %result = alloca i64
  %syncreg = call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.header ], [ %inc.j, %for.j.latch ]
  detach within %syncreg, label %for.j.body, label %for.j.latch

for.j.body:
  call void (i32, ptr, i32, i64, i64, ptr, ...) @llvm.kit.reduce.0(i32 1, ptr %result, i32 8, i64 %j, i64 0, ptr @sum)
  reattach within %syncreg, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !4

for.j.exit:
  sync within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !3

for.i.exit:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1}
!2 = distinct !{!2, !0}
!3 = distinct !{!3}
!4 = distinct !{!4, !0}

; CHECK-DAG: ![[REDUCTION:[0-9]+]] = !{!"tapir.loop.reduction"}
; CHECK-DAG: ![[TARGET:[0-9]+]] = !{!"tapir.loop.target", i32 1}
; CHECK-DAG: ![[ACC_I]] = distinct !{![[ACC_I]]}
; CHECK-DAG: ![[ACC_J]] = distinct !{![[ACC_J]], {{.*}}![[REDUCTION]]{{[,}]}}
; CHECK-DAG: ![[NOACC_I]] = distinct !{![[NOACC_I]]}
; CHECK-DAG: ![[NOACC_J]] = distinct !{![[NOACC_J]],
; CHECK-NOT: ![[REDUCTION]]

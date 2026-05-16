; Check that the kit-annotate-early pass does not add the tapir.loop.reduction
; attribute to non-tapir loops.
;
; RUN: opt -passes="kit-annotate-early" -S %s \
; RUN:     | FileCheck %s
;
; CHECK-NOT: "tapir.loop.reduction"

define void @sum(ptr %res, i64 %v) {
  %1 = load i64, ptr %res
  %2 = add i64 %1, %v
  store i64 %2, ptr %res
  ret void
}

define void @noacc(i64 %n) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i64 [ 0, %for.i.header ], [ %inc.j, %for.j ]
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.i.latch, label %for.j, !llvm.loop !2

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !1

for.i.exit:
  ret void
}

define void @acc(i64 %n) {
entry:
  %result = alloca i64
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i64 [ 0, %for.i.header ], [ %inc.j, %for.j ]
  call void (i32, ptr, i32, i64, i64, ptr, ...) @llvm.kit.reduce.0(i32 1, ptr %result, i32 8, i64 %j, i64 0, ptr @sum)
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.i.latch, label %for.j, !llvm.loop !4

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !3

for.i.exit:
  ret void
}

!1 = distinct !{!1}
!2 = distinct !{!2}
!3 = distinct !{!3}
!4 = distinct !{!4}

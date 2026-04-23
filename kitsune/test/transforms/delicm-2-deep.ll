; Check that the kit-delicm pass sinks instructions into loop nests of depth
; 2 correctly. The tapir loops are nested inside several outer for loops. This
; checks that the implementation of the delicm pass does not assume that the
; depth of the root of the loop nest is 1.
;
; RUN: opt -passes='kit-delicm' -S %s 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @p
; CHECK: header.1:
; CHECK: %iv.1 = phi i64
; CHECK: header.2:
; CHECK: %iv.2 = phi i64
; CHECK: header.3:
; CHECK: %iv.3 = phi i64
; CHECK: for.i.ph:
; CHECK-NEXT: call token @llvm.syncregion.start()
; CHECK-NEXT: br label %for.i.header
; CHECK-EMPTY:
; CHECK-NEXT: for.i.header:
; CHECK-NEXT: %i = phi i64
; CHECK-NEXT: detach
; CHECK-EMPTY:
; CHECK-NEXT: for.i.body:
; CHECK-NEXT: call token @llvm.syncregion.start()
; CHECK-NEXT: br label %for.j.header
; CHECK-EMPTY:
; CHECK-NEXT: for.j.header:
; CHECK-NEXT: %j = phi i64
; CHECK-NEXT: detach
; CHECK-EMPTY:
; CHECK-NEXT: for.j.body:
; CHECK-NEXT: %in = mul i64 %n, %i
; CHECK-NEXT: %in_j = add i64 %in, %j
; CHECK-NEXT: call void @ext(i64 %in_j)
;
define void @p(i64 %m, i64 %n, i64 %p1, i64 %p2, i64 %p3) {
entry:
  br label %header.1

header.1:
  %iv.1 = phi i64 [ 0, %entry ], [ %iv.1.inc, %latch.1 ]
  br label %header.2

header.2:
  %iv.2 = phi i64 [ 0, %header.1 ], [ %iv.2.inc, %latch.2 ]
  br label %header.3

header.3:
  %iv.3 = phi i64 [ 0, %header.2 ], [ %iv.3.inc, %latch.3 ]
  br label %for.i.ph

for.i.ph:
  %syncreg.i = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %for.i.ph ], [ %i.inc, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.j = tail call token @llvm.syncregion.start()
  %in = mul i64 %n, %i
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %j.inc, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  %in_j = add i64 %in, %j
  tail call void @ext(i64 %in_j)
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %j.inc = add i64 %j, 1
  %j.cmp = icmp eq i64 %j.inc, %n
  br i1 %j.cmp, label %for.j.exit, label %for.j.header, !llvm.loop !1

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %i.inc = add i64 %i, 1
  %i.cmp = icmp eq i64 %i, %m
  br i1 %i.cmp, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  br label %latch.3

latch.3:
  %iv.3.inc = add i64 %iv.3, 1
  %cmp.3 = icmp eq i64 %iv.3.inc, %p3
  br i1 %cmp.3, label %latch.2, label %header.3, !llvm.loop !3

latch.2:
  %iv.2.inc = add i64 %iv.2, 1
  %cmp.2 = icmp eq i64 %iv.2.inc, %p2
  br i1 %cmp.2, label %latch.1, label %header.2, !llvm.loop !4

latch.1:
  %iv.1.inc = add i64 %iv.1, 1
  %cmp.1 = icmp eq i64 %iv.1.inc, %p1
  br i1 %cmp.1, label %exit, label %header.1, !llvm.loop !5

exit:
  ret void
}

declare void @ext(i64)

!0 = distinct !{!0, !2}
!1 = distinct !{!1, !2}
!2 = !{!"tapir.loop.target", i32 2}
!3 = distinct !{!3}
!4 = distinct !{!4}
!5 = distinct !{!5}

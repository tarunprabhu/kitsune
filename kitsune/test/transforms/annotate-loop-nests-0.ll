; Check that loop nests that do not contain any tapir loops are not annotated
; with any tapir.loop.nest annotations. The loops here are all perfect.
;
; RUN: opt -passes="kit-annotate-prelower" -S %s \
; RUN:     | FileCheck %s

; CHECK-LABEL: @l1
; CHECK: llvm.loop ![[LOOP1_I:[0-9]+]]
;
; for (i ...)
define void @l1(i64 %n) {
entry:
  %cmp.not = icmp eq i64 %n, 0
  br label %for.i

for.i:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i ]
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i, !llvm.loop !0

for.i.exit:
  ret void
}

; CHECK-LABEL: @l2
; CHECK: llvm.loop ![[LOOP2_J:[0-9]+]]
; CHECK: llvm.loop ![[LOOP2_I:[0-9]+]]
;
; for (i ...)
;   for (j ...)
define void @l2(i64 %n) {
entry:
  %cmp.not = icmp eq i64 %n, 0
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i64 [ 0, %for.i.header ], [ %inc.j, %for.j ]
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.i.latch, label %for.j, !llvm.loop !1

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !2

for.i.exit:
  ret void
}

; CHECK-LABEL: @l3
; CHECK: llvm.loop ![[LOOP3_K:[0-9]+]]
; CHECK: llvm.loop ![[LOOP3_J:[0-9]+]]
; CHECK: llvm.loop ![[LOOP3_I:[0-9]+]]
;
; for (i ...)
;   for (j ...)
;     for (k ...)
define void @l3(i64 %n) {
entry:
  %cmp.not = icmp eq i64 %n, 0
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.header ], [ %inc.j, %for.j.latch ]
  br label %for.k

for.k:
  %k = phi i64 [ 0, %for.j.header ], [ %inc.k, %for.k ]
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %n
  br i1 %cmp.k, label %for.j.latch, label %for.k, !llvm.loop !3

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.i.latch, label %for.j.header, !llvm.loop !4

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !5

for.i.exit:
  ret void
}

; CHECK: ![[LOOP1_I]] = distinct !{![[LOOP1_I]]}
; CHECK: ![[LOOP2_J]] = distinct !{![[LOOP2_J]]}
; CHECK: ![[LOOP2_I]] = distinct !{![[LOOP2_I]]}
; CHECK: ![[LOOP3_K]] = distinct !{![[LOOP3_K]]}
; CHECK: ![[LOOP3_J]] = distinct !{![[LOOP3_J]]}
; CHECK: ![[LOOP3_I]] = distinct !{![[LOOP3_I]]}

!0 = distinct !{!0}
!1 = distinct !{!1}
!2 = distinct !{!2}
!3 = distinct !{!3}
!4 = distinct !{!4}
!5 = distinct !{!5}

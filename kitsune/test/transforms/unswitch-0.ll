; Loop unswitching should work as expected on non-tapir loops.
;
; RUN: opt -passes='function(loop(simple-loop-unswitch),simplifycfg)' -S %s \
; RUN:     -enable-nontrivial-unswitch \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f1
; CHECK-SAME: %[[N:[^,]+]]
; CHECK-SAME: %[[OP:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i32 %[[OP]], 1
; CHECK-NEXT: br i1 %[[CMP]], label %[[LOOP_A:.+]], label %[[LOOP_B:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOP_A]]:
; CHECK-NEXT: %[[IA:.+]] = phi i64
; CHECK-SAME: [ %[[INC_A:.+]], %[[LOOP_A]] ]
; CHECK-SAME: [ 0, %[[ENTRY]] ]
; CHECK-NEXT: call void @add(i64 %[[IA]])
; CHECK-NEXT: %[[INC_A:.+]] = add i64 %[[IA]], 1
; CHECK-NEXT: %[[CMP_A:.+]] = icmp eq i64 %[[INC_A]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP_A]], label %[[EXIT:.+]], label %[[LOOP_A]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOP_B]]:
; CHECK-NEXT: %[[IB:.+]] = phi i64
; CHECK-SAME: [ %[[INC_B:.+]], %[[LOOP_B]] ]
; CHECK-SAME: [ 0, %[[ENTRY]] ]
; CHECK-NEXT: call void @sub(i64 %[[IB]])
; CHECK-NEXT: %[[INC_B:.+]] = add i64 %[[IB]], 1
; CHECK-NEXT: %[[CMP_B:.+]] = icmp eq i64 %[[INC_B]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP_B]], label %[[EXIT]], label %[[LOOP_B]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void

define void @f1(i64 %n, i32 %op) {
entry:
  %cmp.op = icmp eq i32 %op, 1
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  br i1 %cmp.op, label %add, label %sub

add:
  call void @add(i64 %i)
  br label %latch

sub:
  call void @sub(i64 %i)
  br label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %exit, label %header, !llvm.loop !0

exit:
  ret void
}

; CHECK-LABEL: @f2
; CHECK-SAME: %[[M:[^,]+]]
; CHECK-SAME: %[[N:[^,]+]]
; CHECK-SAME: %[[OP:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: br label %[[HEADER_I:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER_I]]:
; CHECK-NEXT: %[[I:.+]] = phi i64
; CHECK-SAME: [ 0, %[[ENTRY]] ]
; CHECK-SAME: [ %[[INC_I:.+]], %[[LATCH_I:.+]] ]
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i32 %[[OP]], 1
; CHECK-NEXT: br i1 %[[CMP]], label %[[LOOP_A:.+]], label %[[LOOP_B:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOP_A]]:
; CHECK-NEXT: %[[IA:.+]] = phi i64
; CHECK-SAME: [ %[[INC_A:.+]], %[[LOOP_A]] ]
; CHECK-SAME: [ 0, %[[HEADER_I]] ]
; CHECK-NEXT: call void @add(i64 %[[IA]])
; CHECK-NEXT: %[[INC_A:.+]] = add i64 %[[IA]], 1
; CHECK-NEXT: %[[CMP_A:.+]] = icmp eq i64 %[[INC_A]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP_A]], label %[[LATCH_I:.+]], label %[[LOOP_A]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOP_B]]:
; CHECK-NEXT: %[[IB:.+]] = phi i64
; CHECK-SAME: [ %[[INC_B:.+]], %[[LOOP_B]] ]
; CHECK-SAME: [ 0, %[[HEADER_I]] ]
; CHECK-NEXT: call void @sub(i64 %[[IB]])
; CHECK-NEXT: %[[INC_B:.+]] = add i64 %[[IB]], 1
; CHECK-NEXT: %[[CMP_B:.+]] = icmp eq i64 %[[INC_B]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP_B]], label %[[LATCH_I]], label %[[LOOP_B]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH_I]]:
; CHECK-NEXT: %[[INC_I]] = add i64 %[[I]], 1
; CHECK-NEXT: %[[CMP_I:.+]] = icmp eq i64 %[[INC_I]], %[[M]]
; CHECK-NEXT: br i1 %[[CMP_I]], label %[[EXIT:.+]], label %[[HEADER_I]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void

define void @f2(i64 %m, i64 %n, i32 %op) {
entry:
  br label %header.i

header.i:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch.i ]
  %cmp.op = icmp eq i32 %op, 1
  br label %header.j

header.j:
  %j = phi i64 [ 0, %header.i ], [ %inc.j, %latch.j ]
  br i1 %cmp.op, label %add, label %sub

add:
  call void @add(i64 %j)
  br label %latch.j

sub:
  call void @sub(i64 %j)
  br label %latch.j

latch.j:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %latch.i, label %header.j

latch.i:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %exit, label %header.i

exit:
  ret void
}

declare void @add(i64)
declare void @sub(i64)

!0 = distinct !{!0}

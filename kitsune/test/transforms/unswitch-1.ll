; Loop unswitching should work as expected on top-level tapir loops.
;
; RUN: opt -passes='function(loop(simple-loop-unswitch),simplifycfg)' -S %s \
; RUN:     -enable-nontrivial-unswitch \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: %[[N:[^,]+]]
; CHECK-SAME: %[[OP:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SR:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i32 %[[OP]], 1
; CHECK-NEXT: br i1 %[[CMP]], label %[[HEADER_A:.+]], label %[[HEADER_B:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER_A]]:
; CHECK-NEXT: %[[IA:.+]] = phi i64
; CHECK-SAME: [ %[[INC_A:[^,]+]], %[[LATCH_A:[^ ]+]] ]
; CHECK-SAME: [ 0, %[[ENTRY]] ]
; CHECK-NEXT: detach within %[[SR]], label %[[BODY_A:.+]], label %[[LATCH_A]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY_A]]:
; CHECK-NEXT: call void @add(i64 %[[IA]])
; CHECK-NEXT: reattach within %[[SR]], label %[[LATCH_A]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH_A]]:
; CHECK-NEXT: %[[INC_A:.+]] = add i64 %[[IA]], 1
; CHECK-NEXT: %[[CMP_A:.+]] = icmp eq i64 %[[INC_A]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP_A]], label %[[SYNC:.+]], label %[[HEADER_A]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER_B]]:
; CHECK-NEXT: %[[IB:.+]] = phi i64
; CHECK-SAME: [ %[[INC_B:[^,]+]], %[[LATCH_B:[^ ]+]] ]
; CHECK-SAME: [ 0, %[[ENTRY]] ]
; CHECK-NEXT: detach within %[[SR]], label %[[BODY_B:.+]], label %[[LATCH_B]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY_B]]:
; CHECK-NEXT: call void @sub(i64 %[[IB]])
; CHECK-NEXT: reattach within %[[SR]], label %[[LATCH_B]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH_B]]:
; CHECK-NEXT: %[[INC_B:.+]] = add i64 %[[IB]], 1
; CHECK-NEXT: %[[CMP_B:.+]] = icmp eq i64 %[[INC_B]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP_B]], label %[[SYNC]], label %[[HEADER_B]]
; CHECK-EMPTY:
; CHECK-NEXT: [[SYNC]]:
; CHECK-NEXT: sync within %[[SR]], label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void

define void @f(i64 %n, i32 %op) {
entry:
  %sr = call token @llvm.syncregion.start()
  %cmp.op = icmp eq i32 %op, 1
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  br i1 %cmp.op, label %add, label %sub

add:
  call void @add(i64 %i)
  br label %reattach

sub:
  call void @sub(i64 %i)
  br label %reattach

reattach:
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %sync, label %header, !llvm.loop !0

sync:
  sync within %sr, label %exit

exit:
  ret void
}

declare void @add(i64)
declare void @sub(i64)

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.target", i32 1}
!2 = !{!"tapir.loop.perfect.depth", i32 1}
!3 = !{!"tapir.loop.perfect.level", i32 1}

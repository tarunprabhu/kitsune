; Loop unswitching ignores top-level tapir loops of depth 1.
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
; CHECK-NEXT: %[[CMPOP:.+]] = icmp eq i32 %[[OP]], 1
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: %[[I:.+]] = phi i64
; CHECK-SAME: [ 0, %[[ENTRY]] ]
; CHECK-SAME: [ %[[INC:.+]], %[[LATCH:.+]] ]
; CHECK-NEXT: detach within %[[SR]], label %[[BODY:.+]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: br i1 %[[CMPOP]], label %[[ADD:.+]], label %[[SUB:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ADD]]:
; CHECK-NEXT: call void @add(i64 %[[I]])
; CHECK-NEXT: br label %[[REATTACH:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[SUB]]:
; CHECK-NEXT: call void @sub(i64 %[[I]])
; CHECK-NEXT: br label %[[REATTACH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH]]:
; CHECK-NEXT: reattach within %[[SR]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH]]:
; CHECK-NEXT: %[[INC:.+]] = add i64 %[[I]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[INC]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[SYNC:.+]], label %[[HEADER]]
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

; Loop unswitching should not run on tapir loops that are not top-level tapir
; loops.
;
; RUN: opt -passes='function(loop(simple-loop-unswitch),simplifycfg)' -S %s \
; RUN:     -enable-nontrivial-unswitch \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: %[[M:[^,]+]]
; CHECK-SAME: %[[N:[^,]+]]
; CHECK-SAME: %[[OP:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SR_I:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: br label %[[HEADER_I:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER_I]]:
; CHECK-NEXT: %[[I:.+]] = phi i64
; CHECK-SAME: [ 0, %[[ENTRY]] ]
; CHECK-SAME: [ %[[INC_I:.+]], %[[LATCH_I:.+]] ]
; CHECK-NEXT: detach within %[[SR_I]], label %[[BODY_I:.+]], label %[[LATCH_I]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY_I]]:
; CHECK-NEXT: %[[SR_J:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i32 %[[OP]], 1
; CHECK-NEXT: br label %[[HEADER_J:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER_J]]:
; CHECK-NEXT: %[[J:.+]] = phi i64
; CHECK-SAME: [ 0, %[[BODY_I]] ]
; CHECK-SAME: [ %[[INC_J:.+]], %[[LATCH_J:.+]] ]
; CHECK-NEXT: detach within %[[SR_J]], label %[[BODY_J:.+]], label %[[LATCH_J]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY_J]]:
; CHECK-NEXT: br i1 %[[CMP]], label %[[ADD:.+]], label %[[SUB:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ADD]]:
; CHECK-NEXT: call void @add(i64 %[[I]], i64 %[[J]])
; CHECK-NEXT: br label %[[REATTACH_J:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[SUB]]:
; CHECK-NEXT: call void @sub(i64 %[[I]], i64 %[[J]])
; CHECK-NEXT: br label %[[REATTACH_J]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH_J]]:
; CHECK-NEXT: reattach within %[[SR_J]], label %[[LATCH_J]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH_J]]:
; CHECK-NEXT: %[[INC_J:.+]] = add i64 %[[J]], 1
; CHECK-NEXT: %[[CMP_J:.+]] = icmp eq i64 %[[INC_J]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP_J]], label %[[SYNC_J:.+]], label %[[HEADER_J]]
; CHECK-EMPTY:
; CHECK-NEXT: [[SYNC_J]]:
; CHECK-NEXT: sync within %[[SR_J]], label %[[REATTACH_I:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH_I]]:
; CHECK-NEXT: reattach within %[[SR_I]], label %[[LATCH_I]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH_I]]:
; CHECK-NEXT: %[[INC_I]] = add i64 %[[I]], 1
; CHECK-NEXT: %[[CMP_I:.+]] = icmp eq i64 %[[INC_I]], %[[M]]
; CHECK-NEXT: br i1 %[[CMP_I]], label %[[SYNC_I:.+]], label %[[HEADER_I]]
; CHECK-EMPTY:
; CHECK-NEXT: [[SYNC_I]]:
; CHECK-NEXT: sync within %[[SR_I]], label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void

define void @f(i64 %m, i64 %n, i32 %op) {
entry:
  %sr.i = call token @llvm.syncregion.start()
  br label %header.i

header.i:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch.i ]
  detach within %sr.i, label %body.i, label %latch.i

body.i:
  %sr.j = call token @llvm.syncregion.start()
  %cmp.op = icmp eq i32 %op, 1
  br label %header.j

header.j:
  %j = phi i64 [ 0, %body.i ], [ %inc.j, %latch.j ]
  detach within %sr.j, label %body.j, label %latch.j

body.j:
  br i1 %cmp.op, label %add, label %sub

add:
  call void @add(i64 %i, i64 %j)
  br label %reattach.j

sub:
  call void @sub(i64 %i, i64 %j)
  br label %reattach.j

reattach.j:
  reattach within %sr.j, label %latch.j

latch.j:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %sync.j, label %header.j, !llvm.loop !1

sync.j:
  sync within %sr.j, label %reattach.i

reattach.i:
  reattach within %sr.i, label %latch.i

latch.i:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %sync.i, label %header.i, !llvm.loop !0

sync.i:
  sync within %sr.i, label %exit

exit:
  ret void
}

declare void @add(i64, i64)
declare void @sub(i64, i64)

!0 = distinct !{!0, !2, !3}
!1 = distinct !{!1, !2, !4, !5}
!2 = !{!"tapir.loop.target", i32 1}
!3 = !{!"tapir.loop.perfect.level", i32 2}
!4 = !{!"tapir.loop.perfect.level", i32 1}
!5 = !{!"tapir.loop.perfect.depth", i32 2}

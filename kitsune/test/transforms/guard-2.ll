; When rotating tapir loops, a guard is always added. The condition of the
; guard is always wrapped in a call to kit.identity. Here, we have a tapir loop
; nest of depth 2 where one of the loops strictly requires a guard and the other
; does not.
;
; RUN: opt -passes=loop-rotate -S %s | FileCheck %s

; CHECK-LABEL: @f
; CHECK-SAME: i64 %[[N1:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SR1:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: %[[CMPN1:.+]] = icmp ult i64 7, %[[N1]]
; CHECK-NEXT: %[[ID1:.+]] = call i1 @llvm.kit.identity.i1(i1 %[[CMPN1]])
; CHECK-NEXT: br i1 %[[ID1]], label %[[PH1:.+]], label %[[SYNC1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH1]]:
; CHECK-NEXT: br label %[[HEADER1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER1]]:
; CHECK-NEXT: %[[I1:.+]] = phi i64
; CHECK-SAME: [ 7, %[[PH1]] ]
; CHECK-SAME: [ %[[INC1:.+]], %[[LATCH1:.+]] ]
; CHECK-NEXT: detach within %[[SR1]], label %[[BODY1:.+]], label %[[LATCH1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY1]]:
; CHECK-NEXT: %[[SR2:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: %[[ID2:.+]] = call i1 @llvm.kit.identity.i1(i1 true)
; CHECK-NEXT: br i1 %[[ID2]], label %[[PH2:.+]], label %[[SYNC2:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH2]]:
; CHECK-NEXT: br label %[[HEADER2:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER2]]:
; CHECK-NEXT: %[[I2:.+]] = phi i64
; CHECK-SAME: [ 3, %[[PH2]] ]
; CHECK-SAME: [ %[[INC2:.+]], %[[LATCH2:.+]] ]
; CHECK-NEXT: detach within %[[SR2]], label %[[BODY2:.+]], label %[[LATCH2]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY2]]:
; CHECK-NEXT: reattach within %[[SR2]], label %[[LATCH2]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH2]]:
; CHECK-NEXT: %[[INC2:.+]] = add i64 %[[I2]], 2
; CHECK-NEXT: %[[CMP2:.+]] = icmp slt i64 %[[INC2]], 31
; CHECK-NEXT: br i1 %[[CMP2]], label %[[HEADER2]], label %[[LOOPEXIT2:.+]],
; CHECK-SAME: !llvm.loop ![[LOOP2:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOPEXIT2]]:
; CHECK-NEXT: br label %[[SYNC2]]
; CHECK-EMPTY:
; CHECK-NEXT: [[SYNC2]]:
; CHECK-NEXT: sync within %[[SR2]], label %[[REATTACH1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH1]]:
; CHECK-NEXT: reattach within %[[SR1]], label %[[LATCH1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH1]]:
; CHECK-NEXT: %[[INC1:.+]] = add i64 %[[I1]], 4
; CHECK-NEXT: %[[CMP1:.+]] = icmp ult i64 %[[INC1]], %[[N1]]
; CHECK-NEXT: br i1 %[[CMP1]], label %[[HEADER1]], label %[[LOOPEXIT1:.+]],
; CHECK-SAME: !llvm.loop ![[LOOP1:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOPEXIT1]]:
; CHECK-NEXT: br label %[[SYNC1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[SYNC1]]:
; CHECK-NEXT: sync within %[[SR1]], label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void
;
; CHECK-DAG: ![[TARGET:[0-9]+]] = !{!"tapir.loop.target", i32 1}
; CHECK-DAG: ![[D:[0-9]+]] = !{!"tapir.loop.perfect.depth", i32 2}
; CHECK-DAG: ![[L1:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 1}
; CHECK-DAG: ![[L2:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 2}
;
; CHECK-DAG: ![[LOOP1]] = distinct !{![[LOOP1]], ![[TARGET]], ![[D]], ![[L1]]}
; CHECK-DAG: ![[LOOP2]] = distinct !{![[LOOP2]], ![[TARGET]], ![[L2]]}

define void @f(i64 %n1) {
ph1:
  %sr1 = call token @llvm.syncregion.start()
  br label %header1

header1:
  %i1 = phi i64 [ 7, %ph1 ], [ %inc1, %latch1 ]
  %cmp1 = icmp ult i64 %i1, %n1
  br i1 %cmp1, label %detach1, label %sync1

detach1:
  detach within %sr1, label %ph2, label %latch1

ph2:
  %sr2 = call token @llvm.syncregion.start()
  br label %header2

header2:
  %i2 = phi i64 [ 3, %ph2 ], [ %inc2, %latch2 ]
  %cmp2 = icmp slt i64 %i2, 31
  br i1 %cmp2, label %detach2, label %sync2

detach2:
  detach within %sr2, label %reattach2, label %latch2

reattach2:
  reattach within %sr2, label %latch2

latch2:
  %inc2 = add i64 %i2, 2
  br label %header2, !llvm.loop !5

sync2:
  sync within %sr2, label %reattach1

reattach1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 4
  br label %header1, !llvm.loop !4

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1}
!1 = !{!"tapir.loop.perfect.depth", i32 2}
!2 = !{!"tapir.loop.perfect.level", i32 1}
!3 = !{!"tapir.loop.perfect.level", i32 2}
!4 = distinct !{!4, !0, !1, !2}
!5 = distinct !{!5, !0, !3}

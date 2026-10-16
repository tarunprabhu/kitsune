; When rotating tapir loops, a guard is always added. The condition of the
; guard is always wrapped in a call to kit.identity. Here, we have a tapir loop
; nest of depth 2 where one of the loops strictly requires a guard and the other
; does not.
;
; RUN: opt -passes=loop-rotate -S %s | FileCheck %s

; CHECK-LABEL: @f
; CHECK-SAME: i64 %[[N1:[^,]+]]
; CHECK-SAME: i64 %[[N3:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SR1:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: %[[SR2:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: %[[SR3:.+]] = call token @llvm.syncregion.start()
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
; CHECK-NEXT: %[[CMPN3:.+]] = icmp slt i64 11, %[[N3]]
; CHECK-NEXT: %[[ID3:.+]] = call i1 @llvm.kit.identity.i1(i1 %[[CMPN3]])
; CHECK-NEXT: br i1 %[[ID3]], label %[[PH3:.+]], label %[[SYNC3:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH3]]:
; CHECK-NEXT: br label %[[HEADER3:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER3]]:
; CHECK-NEXT: %[[I3:.+]] = phi i64
; CHECK-SAME: [ 11, %[[PH3]] ]
; CHECK-SAME: [ %[[INC3:.+]], %[[LATCH3:.+]] ]
; CHECK-NEXT: detach within %[[SR3]], label %[[BODY3:.+]], label %[[LATCH3]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY3]]:
; CHECK-NEXT: reattach within %[[SR3]], label %[[LATCH3]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH3]]:
; CHECK-NEXT: %[[INC3:.+]] = add i64 %[[I3]], 9
; CHECK-NEXT: %[[CMP3:.+]] = icmp slt i64 %[[INC3]], %[[N3]]
; CHECK-NEXT: br i1 %[[CMP3]], label %[[HEADER3]], label %[[LOOPEXIT3:.+]],
; CHECK-SAME: !llvm.loop ![[LOOP3:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOPEXIT3]]:
; CHECK-NEXT: br label %[[SYNC3]]
; CHECK-EMPTY:
; CHECK-NEXT: [[SYNC3]]:
; CHECK-NEXT: sync within %[[SR3]], label %[[REATTACH2:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH2]]:
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
; CHECK-DAG: ![[D:[0-9]+]] = !{!"tapir.loop.perfect.depth", i32 3}
; CHECK-DAG: ![[L1:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 1}
; CHECK-DAG: ![[L2:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 2}
; CHECK-DAG: ![[L3:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 3}
;
; CHECK-DAG: ![[LOOP1]] = distinct !{![[LOOP1]], ![[TARGET]], ![[D]], ![[L1]]}
; CHECK-DAG: ![[LOOP2]] = distinct !{![[LOOP2]], ![[TARGET]], ![[L2]]}
; CHECK-DAG: ![[LOOP3]] = distinct !{![[LOOP3]], ![[TARGET]], ![[L3]]}

define void @f(i64 %n1, i64 %n3) {
entry:
  %sr1 = call token @llvm.syncregion.start()
  %sr2 = call token @llvm.syncregion.start()
  %sr3 = call token @llvm.syncregion.start()
  br label %header1

header1:
  %i1 = phi i64 [ 7, %entry ], [ %inc1, %latch1 ]
  %cmp1 = icmp ult i64 %i1, %n1
  br i1 %cmp1, label %detach1, label %sync1

detach1:
  detach within %sr1, label %header2, label %latch1

header2:
  %i2 = phi i64 [ 3, %detach1 ], [ %inc2, %latch2 ]
  %cmp2 = icmp slt i64 %i2, 31
  br i1 %cmp2, label %detach2, label %sync2

detach2:
  detach within %sr2, label %header3, label %latch2

header3:
  %i3 = phi i64 [ 11, %detach2 ], [ %inc3, %latch3 ]
  %cmp3 = icmp slt i64 %i3, %n3
  br i1 %cmp3, label %detach3, label %sync3

detach3:
  detach within %sr3, label %reattach3, label %latch3

reattach3:
  reattach within %sr3, label %latch3

latch3:
  %inc3 = add i64 %i3, 9
  br label %header3, !llvm.loop !7

sync3:
  sync within %sr3, label %reattach2

reattach2:
  reattach within %sr2, label %latch2

latch2:
  %inc2 = add i64 %i2, 2
  br label %header2, !llvm.loop !6

sync2:
  sync within %sr2, label %reattach1

reattach1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 4
  br label %header1, !llvm.loop !5

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1}
!1 = !{!"tapir.loop.perfect.depth", i32 3}
!2 = !{!"tapir.loop.perfect.level", i32 1}
!3 = !{!"tapir.loop.perfect.level", i32 2}
!4 = !{!"tapir.loop.perfect.level", i32 3}
!5 = distinct !{!5, !0, !1, !2}
!6 = distinct !{!6, !0, !3}
!7 = distinct !{!7, !0, !4}

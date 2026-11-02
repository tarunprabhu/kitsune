; Check that a simple tapir loop of depth 3 with the cuda tapir target is
; prepared as expected.
;
; RUN: opt -passes=kit-prepare -S %s | FileCheck %s

; CHECK-LABEL: @f
; CHECK-SAME: i64 %[[N1:[^,]+]]
; CHECK-SAME: i64 %[[N2:[^,]+]]
; CHECK-SAME: i64 %[[N3:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SR1:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: %[[CMPN1:.+]] = icmp ult i64 0, %[[N1]]
; CHECK-NEXT: br i1 %[[CMPN1]], label %[[PH1:.+]], label %[[SYNC1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH1]]:
; CHECK-NEXT: br label %[[HEADER1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER1]]:
; CHECK-NEXT: %[[I1:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PH1]] ]
; CHECK-SAME: [ %[[INC1:.+]], %[[LATCH1:.+]] ]
; CHECK-NEXT: detach within %[[SR1]], label %[[BODY1:.+]], label %[[LATCH1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY1]]:
; CHECK-NEXT: %[[SR2:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: %[[CMPN2:.+]] = icmp ult i64 0, %[[N2]]
; CHECK-NEXT: br i1 %[[CMPN2]], label %[[PH2:.+]], label %[[SYNC2:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH2]]:
; CHECK-NEXT: br label %[[HEADER2:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER2]]:
; CHECK-NEXT: %[[I2:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PH2]] ]
; CHECK-SAME: [ %[[INC2:.+]], %[[LATCH2:.+]] ]
; CHECK-NEXT: detach within %[[SR2]], label %[[BODY2:.+]], label %[[LATCH2]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY2]]:
; CHECK-NEXT: %[[SR3:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: %[[CMPN3:.+]] = icmp ult i64 0, %[[N3]]
; CHECK-NEXT: br i1 %[[CMPN3]], label %[[PH3:.+]], label %[[SYNC3:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH3]]:
; CHECK-NEXT: br label %[[HEADER3:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER3]]:
; CHECK-NEXT: %[[I3:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PH3]] ]
; CHECK-SAME: [ %[[INC3:.+]], %[[LATCH3:.+]] ]
; CHECK-NEXT: detach within %[[SR3]], label %[[BODY3:.+]], label %[[LATCH3]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY3]]:
; CHECK-NEXT: reattach within %[[SR3]], label %[[LATCH3]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH3]]:
; CHECK-NEXT: %[[INC3]] = add i64 %[[I3]], 1
; CHECK-NEXT: %[[CMP3:.+]] = icmp eq i64 %[[INC3]], %[[N3]]
; CHECK-NEXT: br i1 %[[CMP3]], label %[[LOOPEXIT3:.+]], label %[[HEADER3]]
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
; CHECK-NEXT: %[[INC2]] = add i64 %[[I2]], 1
; CHECK-NEXT: %[[CMP2:.+]] = icmp eq i64 %[[INC2]], %[[N2]]
; CHECK-NEXT: br i1 %[[CMP2]], label %[[LOOPEXIT2:.+]], label %[[HEADER2]]
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
; CHECK-NEXT: %[[INC1:.+]] = add i64 %[[I1]], 1
; CHECK-NEXT: %[[CMP1:.+]] = icmp eq i64 %[[INC1]], %[[N1]]
; CHECK-NEXT: br i1 %[[CMP1]], label %[[LOOPEXIT1:.+]], label %[[HEADER1]]
; CHECK-SAME: !llvm.loop ![[LOOP1:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOPEXIT1]]:
; CHECK-NEXT: br label %[[SYNC1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[SYNC1]]
; CHECK-NEXT: sync within %[[SR1]], label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void
;
; CHECK-DAG: ![[PREPARED:.+]] = !{!"tapir.loop.prepared"}
; CHECK-DAG: ![[TARGET:.+]] = !{!"tapir.loop.target", i32 2}
;
; CHECK-DAG: ![[LOOP1]] = distinct !{![[LOOP1]], ![[TARGET]], {{.+}}, ![[PREPARED]]}
; CHECK-DAG: ![[LOOP2]] = distinct !{![[LOOP2]], ![[TARGET]], {{.+}}, ![[PREPARED]]}
; CHECK-DAG: ![[LOOP3]] = distinct !{![[LOOP3]], ![[TARGET]], {{.+}}, ![[PREPARED]]}

define void @f(i64 %n1, i64 %n2, i64 %n3) {
entry1:
  %sr1 = call token @llvm.syncregion.start()
  %cmpn1 = icmp ult i64 0, %n1
  br i1 %cmpn1, label %ph1, label %sync1

ph1:
  br label %header1

header1:
  %i1 = phi i64 [ 0, %ph1 ], [ %inc1, %latch1 ]
  detach within %sr1, label %body1, label %latch1

body1:
  %sr2 = call token @llvm.syncregion.start()
  %cmpn2 = icmp ult i64 0, %n2
  br i1 %cmpn2, label %ph2, label %sync2

ph2:
  br label %header2

header2:
  %i2 = phi i64 [ 0, %ph2 ], [ %inc2, %latch2 ]
  detach within %sr2, label %body2, label %latch2

body2:
  %sr3 = call token @llvm.syncregion.start()
  %cmpn3 = icmp ult i64 0, %n3
  br i1 %cmpn3, label %ph3, label %sync3

ph3:
  br label %header3

header3:
  %i3 = phi i64 [ 0, %ph3 ], [ %inc3, %latch3 ]
  detach within %sr3, label %body3, label %latch3

body3:
  reattach within %sr3, label %latch3

latch3:
  %inc3 = add i64 %i3, 1
  %cmp3 = icmp eq i64 %inc3, %n3
  br i1 %cmp3, label %loopexit3, label %header3, !llvm.loop !7

loopexit3:
  br label %sync3

sync3:
  sync within %sr3, label %reattach2

reattach2:
  reattach within %sr2, label %latch2

latch2:
  %inc2 = add i64 %i2, 1
  %cmp2 = icmp eq i64 %inc2, %n2
  br i1 %cmp2, label %loopexit2, label %header2, !llvm.loop !6

loopexit2:
  br label %sync2

sync2:
  sync within %sr2, label %reattach1

reattach1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n1
  br i1 %cmp1, label %loopexit1, label %header1, !llvm.loop !5

loopexit1:
  br label %sync1

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

!0 = !{!"tapir.loop.target", i32 2}
!1 = !{!"tapir.loop.perfect.depth", i32 3}
!2 = !{!"tapir.loop.perfect.level", i32 1}
!3 = !{!"tapir.loop.perfect.level", i32 2}
!4 = !{!"tapir.loop.perfect.level", i32 3}
!5 = distinct !{!5, !0, !1, !2}
!6 = distinct !{!6, !0, !3}
!7 = distinct !{!7, !0, !4}

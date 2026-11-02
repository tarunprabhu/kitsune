; Check that a tapir loop nest of depth 2 with the serial tapir target is
; prepared as expected. Currently, we simply serialize all tapir loops in the
; nest.
;
; RUN: opt -passes=kit-prepare -S %s | FileCheck %s

; CHECK-LABEL: @f
; CHECK-SAME: i64 %[[N1:[^,]+]]
; CHECK-SAME: i64 %[[N2:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[CMPN1:.+]] = icmp ult i64 0, %[[N1]]
; CHECK-NEXT: br i1 %[[CMPN1]], label %[[PH1:.+]], label %[[BYPASS1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH1]]:
; CHECK-NEXT: br label %[[HEADER1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER1]]:
; CHECK-NEXT: %[[I1:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PH1]] ]
; CHECK-SAME: [ %[[INC1:.+]], %[[LATCH1:.+]] ]
; CHECK-NEXT: br label %[[BODY1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY1]]:
; CHECK-NEXT: %[[CMPN2:.+]] = icmp ult i64 0, %[[N2]]
; CHECK-NEXT: br i1 %[[CMPN2]], label %[[PH2:.+]], label %[[BYPASS2:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH2]]:
; CHECK-NEXT: br label %[[HEADER2:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER2]]:
; CHECK-NEXT: %[[I2:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PH2]] ]
; CHECK-SAME: [ %[[INC2:.+]], %[[LATCH2:.+]] ]
; CHECK-NEXT: br label %[[BODY2:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY2]]:
; CHECK-NEXT: br label %[[LATCH2]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH2]]:
; CHECK-NEXT: %[[INC2:.+]] = add i64 %[[I2]], 1
; CHECK-NEXT: %[[CMP2:.+]] = icmp eq i64 %[[INC2]], %[[N2]]
; CHECK-NEXT: br i1 %[[CMP2]], label %[[LOOPEXIT2:.+]], label %[[HEADER2]]
; CHECK-SAME: !llvm.loop ![[LOOP2:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOPEXIT2]]:
; CHECK-NEXT: br label %[[BYPASS2]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BYPASS2]]:
; CHECK-NEXT: br label %[[REATTACH1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH1]]:
; CHECK-NEXT: br label %[[LATCH1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH1]]:
; CHECK-NEXT: %[[INC1:.+]] = add i64 %[[I1]], 1
; CHECK-NEXT: %[[CMP1:.+]] = icmp eq i64 %[[INC1]], %[[N1]]
; CHECK-NEXT: br i1 %[[CMP1]], label %[[LOOPEXIT1:.+]], label %[[HEADER1]]
; CHECK-SAME: !llvm.loop ![[LOOP1:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOPEXIT1]]:
; CHECK-NEXT: br label %[[BYPASS1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BYPASS1]]:
; CHECK-NEXT: br label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]
; CHECK-NEXT: ret void

; CHECK-DAG: ![[LOOP1]] = distinct !{![[LOOP1]]}
; CHECK-DAG: ![[LOOP2]] = distinct !{![[LOOP2]]}

define void @f(i64 %n1, i64 %n2) {
entry:
  %sr1 = call token @llvm.syncregion.start()
  %cmp.n1 = icmp ult i64 0, %n1
  br i1 %cmp.n1, label %ph1, label %sync1

ph1:
  br label %header1

header1:
  %i1 = phi i64 [ 0, %ph1 ], [ %inc1, %latch1 ]
  detach within %sr1, label %body1, label %latch1

body1:
  %sr2 = call token @llvm.syncregion.start()
  %cmp.n2 = icmp ult i64 0, %n2
  br i1 %cmp.n2, label %ph2, label %sync2

ph2:
  br label %header2

header2:
  %i2 = phi i64 [ 0, %ph2 ], [ %inc2, %latch2 ]
  detach within %sr2, label %body2, label %latch2

body2:
  reattach within %sr2, label %latch2

latch2:
  %inc2 = add i64 %i2, 1
  %cmp2 = icmp eq i64 %inc2, %n2
  br i1 %cmp2, label %loopexit2, label %header2, !llvm.loop !4

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

!0 = !{!"tapir.loop.target", i32 1}
!1 = !{!"tapir.loop.perfect.depth", i32 2}
!2 = !{!"tapir.loop.perfect.level", i32 2}
!3 = !{!"tapir.loop.perfect.level", i32 1}
!4 = distinct !{!4, !0, !2}
!5 = distinct !{!5, !0, !1, !3}

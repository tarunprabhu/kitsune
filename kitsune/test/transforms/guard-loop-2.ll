; Check that guards for nested tapir loops are added correctly.
;
; RUN: opt -passes='loop-mssa(kit-loop-guard)' -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SR1:.+]] = call token @llvm.syncregion.start
; CHECK-NEXT: br i1 true, label %[[PH1:.+]], label %[[SYNC1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH1]]:
; CHECK-NEXT: br label %[[HEADER1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER1]]:
; CHECK-NEXT: phi i64
; CHECK-SAME: [ 0, %[[PH1]] ]
; CHECK-SAME: [ %{{.+}}, %[[LATCH1:.+]] ]
; CHECK-NEXT: detach within %[[SR1]], label %[[BODY1:.+]], label %[[LATCH1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY1]]:
; CHECK-NEXT: %[[SR2:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: br i1 true, label %[[PH2:.+]], label %[[SYNC2:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH2]]:
; CHECK-NEXT: br label %[[HEADER2:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER2]]:
; CHECK-NEXT: phi i64
; CHECK-SAME: [ 0, %[[PH2]] ]
; CHECK-SAME: [ %{{.+}}, %[[LATCH2:.+]] ]
; CHECK: [[LATCH2]]
; CHECK: br i1 %{{.+}}, label %[[LOOPEXIT2:.+]], label %[[HEADER2]]
; CHECK-SAME: !llvm.loop
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOPEXIT2]]:
; CHECK-NEXT: br label %[[SYNC2]]
; CHECK-EMPTY:
; CHECK-NEXT: [[SYNC2]]:
; CHECK-NEXT: sync within %[[SR2]]
; CHECK: [[LATCH1]]:
; CHECK: br i1 %{{.+}}, label %[[LOOPEXIT1:.+]], label %[[HEADER1]]
; CHECK-SAME: !llvm.loop
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOPEXIT1]]:
; CHECK-NEXT: br label %[[SYNC1]]
; CHECK: [[SYNC1]]:
; CHECK-NEXT: sync within %[[SR1]]

define void @f(i64 %n1, i64 %n2) {
entry:
  %sr1 = call token @llvm.syncregion.start()
  br label %header1

header1:
  %i1 = phi i64 [ 0, %entry ], [ %inc1, %latch1 ]
  detach within %sr1, label %body1, label %latch1

body1:
  %sr2 = call token @llvm.syncregion.start()
  br label %header2

header2:
  %i2 = phi i64 [ 0, %body1 ], [ %inc2, %latch2 ]
  detach within %sr2, label %body2, label %latch2

body2:
  reattach within %sr2, label %latch2

latch2:
  %inc2 = add i64 %i2, 1
  %cmp2 = icmp eq i64 %inc2, %n2
  br i1 %cmp2, label %sync2, label %header2, !llvm.loop !2

sync2:
  sync within %sr2, label %reattach1

reattach1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n1
  br i1 %cmp1, label %sync1, label %header1, !llvm.loop !1

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}
!2 = distinct !{!2, !0}

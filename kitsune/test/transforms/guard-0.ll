; When rotating non-tapir loops, a guard is added only if needed. If it is
; added, it is not wrapped in a kit.identity call.
;
; RUN: opt -passes=loop-rotate -S %s | FileCheck %s

; CHECK-LABEL: @fnc
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[CMPN:.+]] = icmp ult i64 3, %[[N]]
; CHECK-NEXT: br i1 %[[CMPN]], label %[[PH:.+]], label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH]]:
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: %[[I:.+]] = phi i64
; CHECK-SAME: [ 3, %[[PH]] ]
; CHECK-SAME: [ %[[INC:.+]], %[[LATCH:.+]] ]
; CHECK-NEXT: br label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH]]:
; CHECK-NEXT: %[[INC:.+]] = add i64 %[[I]], 2
; CHECK-NEXT: %[[CMP:.+]] = icmp ult i64 %[[INC]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[HEADER]], label %[[LOOPEXIT:.+]],
; CHECK-SAME: llvm.loop ![[LOOPNC:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOPEXIT]]:
; CHECK-NEXT: br label %[[EXIT]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void
;
define void @fnc(i64 %n) {
entry:
  br label %header

header:
  %i = phi i64 [ 3, %entry ], [ %inc, %latch ]
  %cmp = icmp ult i64 %i, %n
  br i1 %cmp, label %body, label %exit

body:
  br label %latch

latch:
  %inc = add i64 %i, 2
  br label %header, !llvm.loop !0

exit:
  ret void
}

; CHECK-LABEL: @fc
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: %[[I:.+]] = phi i64
; CHECK-SAME: [ 3, %[[ENTRY]] ]
; CHECK-SAME: [ %[[INC:.+]], %[[LATCH:.+]] ]
; CHECK-NEXT: br label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH]]:
; CHECK-NEXT: %[[INC:.+]] = add i64 %[[I]], 2
; CHECK-NEXT: %[[CMP:.+]] = icmp slt i64 %[[INC]], 31
; CHECK-NEXT: br i1 %[[CMP]], label %[[HEADER:.+]], label %[[EXIT:[^,]+]]
; CHECK-SAME: !llvm.loop ![[LOOPC:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void
;
define void @fc() {
entry:
  br label %header

header:
  %i = phi i64 [ 3, %entry ], [ %inc, %latch ]
  %cmp = icmp slt i64 %i, 31
  br i1 %cmp, label %body, label %exit

body:
  br label %latch

latch:
  %inc = add i64 %i, 2
  br label %header, !llvm.loop !1

exit:
  ret void
}

; CHECK-DAG: ![[LOOPC]] = distinct !{![[LOOPC]]}
; CHECK-DAG: ![[LOOPNC]] = distinct !{![[LOOPNC]]}

!0 = distinct !{!0}
!1 = distinct !{!1}

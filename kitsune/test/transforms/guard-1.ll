; When rotating tapir loops, a guard is always added. The condition of the
; guard is always wrapped in a call to kit.identity.
;
; RUN: opt -passes=loop-rotate -S %s | FileCheck %s

; CHECK-LABEL: @fnc
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SR:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: %[[CMPN:.+]] = icmp ult i64 3, %[[N]]
; CHECK-NEXT: %[[ID:.+]] = call i1 @llvm.kit.identity.i1(i1 %[[CMPN]])
; CHECK-NEXT: br i1 %[[ID]], label %[[PH:.+]], label %[[SYNC:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH]]:
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: %[[I:.+]] = phi i64
; CHECK-SAME: [ 3, %[[PH]] ]
; CHECK-SAME: [ %[[INC:.+]], %[[LATCH:.+]] ]
; CHECK-NEXT: detach within %[[SR]], label %[[REATTACH:.+]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH]]:
; CHECK-NEXT: reattach within %[[SR]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH]]:
; CHECK-NEXT: %[[INC:.+]] = add i64 %[[I]], 2
; CHECK-NEXT: %[[CMP:.+]] = icmp ult i64 %[[INC]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[HEADER]], label %[[LOOPEXIT:.+]],
; CHECK-SAME: llvm.loop ![[LOOPNC:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOPEXIT]]:
; CHECK-NEXT: br label %[[SYNC]]
; CHECK-EMPTY:
; CHECK-NEXT: [[SYNC]]:
; CHECK-NEXT: sync within %[[SR]], label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void
;
define void @fnc(i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 3, %entry ], [ %inc, %latch ]
  %cmp = icmp ult i64 %i, %n
  br i1 %cmp, label %detach, label %sync

detach:
  detach within %sr, label %reattach, label %latch

reattach:
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 2
  br label %header, !llvm.loop !1

sync:
  sync within %sr, label %exit

exit:
  ret void
}

; CHECK-LABEL: @fc
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SR:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: %[[ID:.+]] = call i1 @llvm.kit.identity.i1(i1 true)
; CHECK-NEXT: br i1 %[[ID]], label %[[PH:.+]], label %[[SYNC:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH]]:
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: %[[I:.+]] = phi i64
; CHECK-SAME: [ 3, %[[PH]] ]
; CHECK-SAME: [ %[[INC:.+]], %[[LATCH:.+]] ]
; CHECK-NEXT: detach within %[[SR]], label %[[REATTACH:.+]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH]]:
; CHECK-NEXT: reattach within %[[SR]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH]]:
; CHECK-NEXT: %[[INC:.+]] = add i64 %[[I]], 2
; CHECK-NEXT: %[[CMP:.+]] = icmp slt i64 %[[INC]], 31
; CHECK-NEXT: br i1 %[[CMP]], label %[[HEADER:.+]], label %[[LOOPEXIT:[^,]+]]
; CHECK-SAME: !llvm.loop ![[LOOPC:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOPEXIT]]:
; CHECK-NEXT: br label %[[SYNC]]
; CHECK-EMPTY:
; CHECK-NEXT: [[SYNC]]:
; CHECK-NEXT: sync within %[[SR]], label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void
;
define void @fc() {
entry:
  %sr = call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 3, %entry ], [ %inc, %latch ]
  %cmp = icmp slt i64 %i, 31
  br i1 %cmp, label %detach, label %sync

detach:
  detach within %sr, label %reattach, label %latch

reattach:
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 2
  br label %header, !llvm.loop !2

sync:
  sync within %sr, label %exit

exit:
  ret void
}

; CHECK-DAG: ![[TARGET:[0-9]+]] = !{!"tapir.loop.target", i32 1}
; CHECK-DAG: ![[LOOPC]] = distinct !{![[LOOPC]], ![[TARGET]]}
; CHECK-DAG: ![[LOOPNC]] = distinct !{![[LOOPNC]], ![[TARGET]]}

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}
!2 = distinct !{!2, !0}

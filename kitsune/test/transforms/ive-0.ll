; The secondary induction variable elimination pass has no effect on non-tapir
; loops and tapir loops that do not contain secondary induction variables.
;
; RUN: opt -passes=kit-ive -S %s | FileCheck %s

; CHECK-LABEL: @normal_loop
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: br label %[[LOOP:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOP]]:
; CHECK-NEXT: %[[IV_I:.+]] = phi i64 [ 0, %[[ENTRY]] ], [ %[[INC_I:.+]], %[[LOOP]] ]
; CHECK-NEXT: %[[IV_J:.+]] = phi i32 [ 99, %[[ENTRY]] ], [ %[[INC_J:.+]], %[[LOOP]] ]
; CHECK-NEXT: call void @ext(i32 %[[IV_J]])
; CHECK-NEXT: %[[INC_I]] = add i64 %[[IV_I]], 1
; CHECK-NEXT: %[[INC_J]] = add i32 %[[IV_J]], 5
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[IV_I]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[LOOP]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void
define void @normal_loop(i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %loop ]
  %j = phi i32 [ 99, %entry ], [ %next.j, %loop ]
  call void @ext(i32 %j)
  %inc.i = add i64 %i, 1
  %next.j = add i32 %j, 5
  %cmp.i = icmp eq i64 %i, %n
  br i1 %cmp.i, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

; CHECK-LABEL: @no_secondary_ivs
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SYNCREG:.+]] = tail call token @llvm.syncregion.start()
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: %[[IV:.+]] = phi i64
; CHECK-SAME: [ 0, %[[ENTRY]] ]
; CHECK-SAME: [ %[[INC:.+]], %[[LATCH:[^ ]+]] ]
; CHECK-NEXT: detach within %[[SYNCREG]], label %[[BODY:.+]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: reattach within %[[SYNCREG]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH]]:
; CHECK-NEXT: %[[INC]] = add i64 %[[IV]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[IV]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[HEADER]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: sync within %[[SYNCREG]], label %[[END:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[END]]:
; CHECK-NEXT: ret void
define void @no_secondary_ivs(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  reattach within %syncreg, label %latch

latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %i, %n
  br i1 %cmp.i, label %exit, label %header, !llvm.loop !1

exit:
  sync within %syncreg, label %end

end:
  ret void
}

declare void @ext(i32)

!0 = distinct !{!0}
!1 = distinct !{!1, !2}
!2 = !{!"tapir.loop.target", i32 1}

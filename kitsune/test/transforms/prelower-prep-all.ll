; If all the non-PHI instructions in the loop header are safe to sink, all must
; be sunk.
;
; RUN: opt -passes=kit-prepare-prelower -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SYNCREG:.+]] = tail call token @llvm.syncregion.start()
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: %[[IV:.+]] = phi i64
; CHECK-NEXT: detach within %[[SYNCREG]], label %[[BODY:.+]], label %[[LATCH:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: %[[TRUNC:.+]] = trunc i64 %[[IV]] to i32
; CHECK-NEXT: %[[PREV:.+]] = sub i32 %[[TRUNC]], 1
; CHECK-NEXT: call void @ext(i32 %[[PREV]])
; CHECK-NEXT: reattach within %[[SYNCREG]]

define void @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch ]
  %trunc.i = trunc i64 %i to i32
  %prev.i = sub i32 %trunc.i, 1
  detach within %syncreg, label %body, label %latch

body:
  call void @ext(i32 %prev.i)
  reattach within %syncreg, label %latch

latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %i, %n
  br i1 %cmp.i, label %exit, label %header, !llvm.loop !0

exit:
  sync within %syncreg, label %end

end:
  ret void
}

declare void @ext(i32)

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 1024}

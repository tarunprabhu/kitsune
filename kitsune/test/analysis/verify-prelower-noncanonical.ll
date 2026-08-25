; Tapir loops must contain a canonical induction variable.
;
; RUN: not opt -passes=kit-verify-prelower %s -disable-output 2>&1 \
; RUN:     | FileCheck %s

; CHECK: tapir loop does not have a canonical induction variable
; CHECK-NEXT: from loop 'p.loop.i'
; CHECK-NEXT: from function 'p'

define void @p(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 10, %entry ], [ %inc.i, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  reattach within %syncreg, label %latch

latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %sync, label %header, !llvm.loop !0

sync:
  sync within %syncreg, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.target", i32 1024}
!2 = !{!"tapir.loop.name", !"p.loop.i"}

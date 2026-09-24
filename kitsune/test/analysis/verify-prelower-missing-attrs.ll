; All tapir loops must have the `perfect.level` attribute. Top-level tapir loops
; must also have the 'perfect.depth' attribute.
;
; RUN: not opt -passes='kit-verify-prelower' -S %s 2>&1 | FileCheck %s

; CHECK: missing required attribute 'tapir.loop.perfect.depth'
; CHECK-NEXT: from loop 'loop.f1'
;
; forall (...)
;   ...
define void @f1(i64 %n) {
entry:
  %syncreg = call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  reattach within %syncreg, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %sync, label %header, !llvm.loop !3

sync:
  sync within %syncreg, label %exit

exit:
  ret void
}

; CHECK: missing required attribute 'tapir.loop.perfect.level'
; CHECK-NEXT: from loop 'loop.f2'
;
; forall (...)
;   ...
define void @f2(i64 %n) {
entry:
  %syncreg = call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  reattach within %syncreg, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %sync, label %header, !llvm.loop !5

sync:
  sync within %syncreg, label %exit

exit:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1}
!1 = !{!"tapir.loop.perfect.level", i32 1}
!2 = !{!"tapir.loop.perfect.depth", i32 1}
!3 = distinct !{!3, !0, !4, !1}
!4 = !{!"tapir.loop.name", !"loop.f1"}
!5 = distinct !{!5, !0, !6, !2}
!6 = !{!"tapir.loop.name", !"loop.f2"}

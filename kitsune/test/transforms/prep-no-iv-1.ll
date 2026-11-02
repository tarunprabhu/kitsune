; All tapir loops in a nest being prepared must have a canonical induction
; variable.
;
; RUN: not opt -passes='kit-prepare' -S %s 2>&1 | FileCheck %s
;
; CHECK: tapir loop does not have a canonical induction variable

define void @acc(double %n) {
entry:
  %sr = tail call token @llvm.syncregion.start()
  %cmpn1 = fcmp ogt double 0.0, %n
  br i1 %cmpn1, label %ph, label %sync

ph:
  br label %header

header:
  %j = phi double [ 0.0, %ph ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  reattach within %sr, label %latch

latch:
  %inc = fadd double %j, 1.0
  %cmp = fcmp oeq double %inc, %n
  br i1 %cmp, label %loopexit, label %header, !llvm.loop !0

loopexit:
  br label %sync

sync:
  sync within %sr, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.target", i32 1024}
!2 = distinct !{!"tapir.loop.perfect.depth", i32 1}
!3 = distinct !{!"tapir.loop.perfect.level", i32 1}

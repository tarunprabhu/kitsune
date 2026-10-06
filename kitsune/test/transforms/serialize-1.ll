; By default, tapir loops of depth 1 should not be serialized. We set the tapir
; target to 'serial' because tapir loops with this tapir target are not treated
; any differently from loops with a parallel tapir target (CPU, or GPU-centric).
;
; RUN: opt -passes=kit-serialize -S %s | FileCheck %s

; CHECK-LABEL: @f
; CHECK: %[[SYNCREG:.+]] = call token @llvm.syncregion.start
; CHECK: detach within %[[SYNCREG]]
; CHECK: reattach within %[[SYNCREG]]

define void @f(i64 %n) !kit.func !0 {
entry:
  %sr = call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %sync, label %header, !llvm.loop !2

sync:
  sync within %sr, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"kit.func.pre.lower.annotate.pass"}
!2 = distinct !{!2, !3, !4, !5}
!3 = !{!"tapir.loop.target", i32 1}
!4 = !{!"tapir.loop.perfect.depth", i32 1}
!5 = !{!"tapir.loop.perfect.level", i32 1}

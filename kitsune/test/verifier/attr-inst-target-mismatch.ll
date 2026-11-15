; If a target attribute is present on an instruction in a tapir loop, that
; target must match that of the containing loop.
;
; RUN: not llvm-as -o /dev/null %s 2>&1 | FileCheck %s
;
; CHECK: target attribute mismatch between instruction ('serial') and containing loop ('openmp')

define void @f(i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch, !kit.inst !1

body:
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %sync, label %header, !llvm.loop !0

sync:
  sync within %sr, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !2}
!1 = distinct !{!1, !3}
!2 = !{!"tapir.loop.target", i32 512}
!3 = !{!"kit.inst.target", i32 1}

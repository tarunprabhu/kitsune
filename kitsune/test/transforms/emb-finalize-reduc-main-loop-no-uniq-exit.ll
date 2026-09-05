; The exit block of the main kernel loop must contain a single return
; instruction.
;
; RUN: %kit-enc %s \
; RUN:     | not opt -passes='emb-finalize-reductions' 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: main kernel loop must have a unique exit block
; CHECK-NEXT: from loop 'kernel.main'

define void @f(i32 %n) !kit.func !0 {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %next ]
  %is42 = icmp eq i32 %i, 42
  br i1 %is42, label %ret, label %next

next:
  %inc = add i32 %i, 1
  %cmp = icmp eq i32 %inc, %n
  br i1 %cmp, label %exit, label %loop, !llvm.loop !2

exit:
  ret void

ret:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"kit.func.kernel", i32 1}
!2 = distinct !{!2, !3}
!3 = !{!"tapir.loop.name", !"kernel.main"}

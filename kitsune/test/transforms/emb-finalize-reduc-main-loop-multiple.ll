; The kernel function must contain exactly one top-level loop.
;
; RUN: %kit-enc %s \
; RUN:     | not opt -passes='emb-finalize-reductions' 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: kernel function must contain exactly one top-level loop

define void @f(i32 %n) !kit.func !0 {
ph1:
  br label %loop1

loop1:
  %i1 = phi i32 [ 0, %ph1 ], [ %inc1, %loop1 ]
  %inc1 = add i32 %i1, 1
  %cmp1 = icmp eq i32 %inc1, %n
  br i1 %cmp1, label %ph2, label %loop1, !llvm.loop !2

ph2:
  br label %loop2

loop2:
  %i2 = phi i32 [ 0, %ph2 ], [ %inc2, %loop2 ]
  %inc2 = add i32 %i2, 1
  %cmp2 = icmp eq i32 %inc2, %n
  br i1 %cmp2, label %exit, label %loop2, !llvm.loop !3

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"kit.func.kernel", i32 1}
!2 = distinct !{!2}
!3 = distinct !{!3}

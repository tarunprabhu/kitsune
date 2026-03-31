; The loop threads-per-block attribute must be in the range [0, 1024]
;
; RUN: llvm-as -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --allow-empty
;
; CHECK-NOT: {{.+}}

define void @lo(i64 %n) {
entry:
  br label %for.lo

for.lo:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.lo ]
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %exit, label %for.lo, !llvm.loop !2

exit:
  ret void
}

define void @hi(i64 %n) {
entry:
  br label %for.hi

for.hi:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.hi ]
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %exit, label %for.hi, !llvm.loop !3

exit:
  ret void
}

!0 = !{!"tapir.loop.threads.per.block", i32 0}
!1 = !{!"tapir.loop.threads.per.block", i32 1024}
!2 = distinct !{!2, !0}
!3 = distinct !{!3, !1}

; The loop threads-per-block attribute must be in the range [0, 1024]
;
; RUN: not llvm-as -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: attribute 'tapir.loop.threads.per.block': invalid value
; CHECK-SAME: Must be in the range [0,1024]
; CHECK-NEXT: from loop 'for.udf'
; CHECK-NEXT: from function 'underflow'
;
; CHECK: attribute 'tapir.loop.threads.per.block': invalid value
; CHECK-SAME: Must be in the range [0,1024]
; CHECK-NEXT: from loop 'for.ovf'
; CHECK-NEXT: from function 'overflow'

define void @underflow(i64 %n) {
entry:
  br label %for.udf

for.udf:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.udf ]
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %exit, label %for.udf, !llvm.loop !2

exit:
  ret void
}

define void @overflow(i64 %n) {
entry:
  br label %for.ovf

for.ovf:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.ovf ]
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %exit, label %for.ovf, !llvm.loop !3

exit:
  ret void
}

!0 = !{!"tapir.loop.threads.per.block", i32 -1}
!1 = !{!"tapir.loop.threads.per.block", i32 1025}
!2 = distinct !{!2, !0}
!3 = distinct !{!3, !1}

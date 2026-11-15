; Detach/reattach instructions without the target attribute are only allowed
; in tapir loops.
;
; RUN: not opt -passes='kit-verify-prelower' %s -disable-output 2>&1 \
; RUN:     | FileCheck %s

; CHECK: tapir instructions outside tapir loops are not yet supported
;
; for (i ... m)
define void @f1(i64 %n) {
entry:
  %sr = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

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

; In this test, the detach instruction is in a loop that does not have the
; "tapir.loop.target" attribute, and is, therefore, not recognized as being a
; tapir loop.
;
; CHECK: tapir instructions outside tapir loops are not yet supported
;
; forall (i ... m)
;   for (j ... n)
;     for (k ... p)
define void @f3(i64 %m, i64 %n, i64 %p) {
entry:
  %sr1 = tail call token @llvm.syncregion.start()
  br label %header1

header1:
  %i = phi i64 [ 0, %entry ], [ %inc1, %latch1 ]
  detach within %sr1, label %body1, label %latch1

body1:
  %sr2 = tail call token @llvm.syncregion.start()
  br label %header2

header2:
  %j = phi i64 [ 0, %body1 ], [ %inc2, %latch2 ]
  detach within %sr2, label %body2, label %latch2

body2:
  br label %body3

body3:
  %k = phi i64 [ 0, %body2 ], [ %inc3, %body3 ]
  %inc3 = add i64 %k, 1
  %cmp3 = icmp eq i64 %inc3, %p
  br i1 %cmp3, label %exit3, label %body3, !llvm.loop !3

exit3:
  reattach within %sr2, label %latch2

latch2:
  %inc2 = add i64 %j, 1
  %cmp2 = icmp eq i64 %inc2, %n
  br i1 %cmp2, label %exit2, label %header2, !llvm.loop !2

exit2:
  sync within %sr2, label %end2

end2:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i, 1
  %cmp1 = icmp eq i64 %inc1, %m
  br i1 %cmp1, label %exit1, label %header1, !llvm.loop !1

exit1:
  sync within %sr1, label %end1

end1:
  ret void
}

!0 = distinct !{!0}
!1 = distinct !{!1, !4}
!2 = distinct !{!2}
!3 = distinct !{!3}
!4 = !{!"tapir.loop.target", i32 1024}

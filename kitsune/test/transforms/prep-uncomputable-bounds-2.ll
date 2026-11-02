; All tapir loops in a nest that are being prepared must have computable loop
; bounds
;
; RUN: not opt -passes='kit-prepare' -S %s 2>&1 | FileCheck %s
;
; CHECK: tapir loop trip count is not finite

define void @f(i64 %n1, i64 %n2) {
entry:
  %sr1 = call token @llvm.syncregion.start()
  %cmp.n1 = icmp slt i64 0, %n1
  br i1 %cmp.n1, label %ph1, label %sync1

ph1:
  br label %header1

header1:
  %i1 = phi i64 [ 0, %ph1 ], [ %inc1, %latch1 ]
  detach within %sr1, label %body1, label %latch1

body1:
  %sr2 = call token @llvm.syncregion.start()
  %cmp.n2 = icmp slt i64 0, %n2
  br i1 %cmp.n2, label %ph2, label %sync2

ph2:
  br label %header2

header2:
  %i2 = phi i64 [ 0, %ph2 ], [ %inc2, %latch2 ]
  detach within %sr2, label %body2, label %latch2

body2:
  reattach within %sr2, label %latch2

latch2:
  %inc2 = add i64 %i2, 1
  %cmp2 = icmp eq i64 %inc2, %i2
  br i1 %cmp2, label %loopexit2, label %header2, !llvm.loop !4

loopexit2:
  br label %sync2

sync2:
  sync within %sr2, label %reattach1

reattach1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n1
  br i1 %cmp1, label %loopexit1, label %header1, !llvm.loop !5

loopexit1:
  br label %sync1

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

!0 = !{!"tapir.loop.target", i32 512}
!1 = !{!"tapir.loop.perfect.depth", i32 2}
!2 = !{!"tapir.loop.perfect.level", i32 2}
!3 = !{!"tapir.loop.perfect.level", i32 1}
!4 = distinct !{!4, !0, !2}
!5 = distinct !{!5, !0, !1, !3}

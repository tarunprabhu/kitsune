; If the final value of the loop induction variable of a tapir loop is defined
; in the loop preheader, the instruction must be cloned into the loop guard
; block. But some instructions are not safe to clone since they may have
; side-effects. We currently do not support this.
;
; RUN: not opt -passes=kit-prepare -S %s 2>&1 | FileCheck %s
;
; CHECK: NOT YET IMPLEMENTED: Expand unsafe final IV value into loop guard block

define void @f(i32 %n1) {
entry:
  %sr1 = call token @llvm.syncregion.start();
  %cmp.n1 = icmp slt i32 2, %n1
  br i1 %cmp.n1, label %ph1, label %sync1

ph1:
  %nminus2 = call i32 @sub2(i32 %n1)
  %tc = zext i32 %nminus2 to i64
  br label %header1

header1:
  %i1 = phi i64 [ 0, %ph1 ], [ %inc1, %latch1 ]
  detach within %sr1, label %body1, label %latch1

body1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %tc
  br i1 %cmp1, label %loopexit1, label %header1, !llvm.loop !3

loopexit1:
  br label %sync1

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

declare i32 @sub2(i32)

!0 = !{!"tapir.loop.target", i32 512}
!1 = !{!"tapir.loop.perfect.depth", i32 1}
!2 = !{!"tapir.loop.perfect.level", i32 1}
!3 = distinct !{!3, !0, !1, !2}

; All tapir loops in a nest that are being prepared must have computable loop
; bounds
;
; RUN: not opt -passes='kit-prepare' -S %s 2>&1 | FileCheck %s
;
; CHECK: tapir loop trip count is not finite

define void @f(ptr %ptrn1) {
entry:
  %sr1 = call token @llvm.syncregion.start();
  %n1.init = load i64, ptr %ptrn1
  %cmpn1 = icmp slt i64 0, %n1.init
  br i1 %cmpn1, label %ph1, label %sync1

ph1:
  br label %header1

header1:
  %i1 = phi i64 [ 0, %ph1 ], [ %inc1, %latch1 ]
  detach within %sr1, label %body1, label %latch1

body1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %n1 = load i64, ptr %ptrn1
  %cmp1 = icmp eq i64 %inc1, %n1
  br i1 %cmp1, label %loopexit1, label %header1, !llvm.loop !3

loopexit1:
  br label %sync1

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1}
!1 = !{!"tapir.loop.perfect.depth", i32 1}
!2 = !{!"tapir.loop.perfect.level", i32 1}
!3 = distinct !{!3, !0, !1, !2}

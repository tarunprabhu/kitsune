; Check that the default iterations per block are set as expected. Currently,
; this defaults to 1.
;
; RUN: opt -passes=kit-gpu-loop-block -S %s | FileCheck %s
;
; @0 = internal constant i64 1
; CHECK: call i64 @llvm.kit.gpu.ipt(i32 2, ptr @0)

@g32 = internal constant i32 1

define void @f(i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  %ipt = call i64 @llvm.kit.gpu.ipt(i32 2, ptr null)
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

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 2}

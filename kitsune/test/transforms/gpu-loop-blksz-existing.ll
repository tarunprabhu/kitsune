; If a global integer of an appropriate type and initializer exists, use it as
; the argument to the `kit.gpu.ipt` intrinsic.
;
; RUN: opt -passes=kit-gpu-loop-block -S %s \
; RUN:     | FileCheck %s --check-prefix=DEFAULT
;
; DEFAULT: call i64 @llvm.kit.gpu.ipt(i32 4, ptr @gd64)
;
; RUN: opt -passes=kit-gpu-loop-block -kit-gpu-loop-ipt=17 -S %s \
; RUN:     | FileCheck %s --check-prefix=EXPLICIT
;
; EXPLICIT: call i64 @llvm.kit.gpu.ipt(i32 4, ptr @ge64)

@gd32 = internal constant i32 1
@gd64 = internal constant i64 1
@ge32 = internal constant i32 17
@ge64 = internal constant i64 17

define void @f(i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  %ipt = call i64 @llvm.kit.gpu.ipt(i32 4, ptr null)
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
!1 = !{!"tapir.loop.target", i32 4}


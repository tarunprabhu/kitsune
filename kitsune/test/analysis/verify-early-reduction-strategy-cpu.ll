; Check that an error is emitted if the reduction strategy is incompatible with
; the tapir target on the loop.
;
; RUN: not opt -passes='kit-verify-early' %s 2>&1 | FileCheck %s
;
; CHECK: invalid reduction strategy 'wshfl' for tapir target 'serial'

define void @fcpu(ptr %res, i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1, i32 1, ptr %res, i64 %i, i32 9, i32 8, i64 0, ptr @sum)
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %exit, label %header, !llvm.loop !2

exit:
  ret void
}

declare void @sum(ptr, i64)

!0 = !{!"tapir.loop.target", i32 1}
!1 = !{!"tapir.loop.reduction"}
!2 = distinct !{!2, !0, !1}

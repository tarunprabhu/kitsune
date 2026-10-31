; The current implementation replaces the "auto" strategy with a default that
; depends on the tapir target on the reduction loop.
;
; If we ever implement a more intelligent approach that determines a strategy
; based on the contents of the loop, this file should be replaced with tests
; that exercise the analysis.
;
; RUN: opt -passes='kit-reduce-strategy' -S %s | FileCheck %s

; CHECK-LABEL: @fcpu
; CHECK-SAME: ptr %[[RES:[^,]+]]
; CHECK: %[[I:.+]] = phi i64
; CHECK: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: ptr %[[RES]], i64 %[[I]], i32 4
define void @fcpu(ptr %res, i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 512, i32 1, ptr %res, i64 %i, i32 0, i32 8, i64 0, ptr @sum)
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %exit, label %header, !llvm.loop !3

exit:
  ret void
}

; CHECK-LABEL: @fgpu
; CHECK-SAME: ptr %[[RES:[^,]+]]
; CHECK: %[[I:.+]] = phi i64
; CHECK: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: ptr %[[RES]], i64 %[[I]], i32 9
define void @fgpu(ptr %res, i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 4, i32 1, ptr %res, i64 %i, i32 0, i32 8, i64 0, ptr @sum)
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %exit, label %header, !llvm.loop !4

exit:
  ret void
}

declare void @sum(ptr, i64)

!0 = !{!"tapir.loop.target", i32 512}
!1 = !{!"tapir.loop.target", i32 4}
!2 = !{!"tapir.loop.reduction"}
!3 = distinct !{!3, !0, !2}
!4 = distinct !{!4, !1, !2}

; By default, the threads per block argument passed to the launch intrinsic
; should be 0.
;
; RUN: opt --tapir=cuda -passes='loop-spawning' -S %s | FileCheck %s
;
; CHECK: define {{.+}} @f(i64 {{.*}}%[[N:.+]])
;
; CHECK: %{{[0-9]+}} = {{.*}}call {{.+}} @llvm.kit.async.gpu.kernel.launch(
; CHECK-SAME: i32 2,
; CHECK-SAME: ptr @{{[^,]+}},
; CHECK-SAME: ptr @{{[^,]+}},
; CHECK-SAME: i64 0,
; CHECK-SAME: i64 0,
; CHECK-SAME: i64 %[[N]],
; CHECK-SAME: i32 0,
; CHECK-SAME: ptr @{{[^,]+}},
; CHECK-SAME: ptr %{{[^,]+}},
; CHECK-SAME: i64 0,
; CHECK-SAME: i64 %[[N]])

declare void @ext(i64)

define void @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  call void @ext(i64 %i)
  reattach within %syncreg, label %latch

latch:
  %i.next = add i64 %i, 1
  %cmp.i = icmp eq i64 %i.next, %n
  br i1 %cmp.i, label %sync, label %header, !llvm.loop !0

sync:
  sync within %syncreg, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"tapir.loop.spawn.strategy", i32 3}
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"tapir.loop.lowering.enabled"}
!4 = !{!"tapir.loop.perfect.depth", i32 1}
!5 = !{!"tapir.loop.perfect.level", i32 1}

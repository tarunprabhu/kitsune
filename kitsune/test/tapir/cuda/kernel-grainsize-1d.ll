; The grainsize is always set to 1 currently. Check that this is the case in
; the kernel immediately after loop spawning.
;
; RUN: opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:     --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:     -passes='loop-spawning' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: define ptx_kernel
; CHECK: [[ENTRY:[^:]+]]:
; CHECK: %[[IV_START:.+]] = zext i32 %{{.+}} to i64
; CHECK: %[[IV_END:.+]] = add {{.*}}i64 %[[IV_START]], 1
; CHECK: %[[IV:.+]] = phi i64
; CHECK-SAME: %[[IV_START]], %[[ENTRY]]
; CHECK: %[[INC:.+]] = add {{.*}}i64 %[[IV]], 1
; CHECK: icmp eq i64 %[[INC]], %[[IV_END]]

define void @p(ptr %a, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %a.i = getelementptr i64, ptr %a, i64 %i
  store i64 %i, ptr %a.i
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

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.target", i32 2}
!2 = !{!"tapir.loop.spawn.strategy", i32 3}
!3 = !{!"tapir.loop.lowerig.enabled"}

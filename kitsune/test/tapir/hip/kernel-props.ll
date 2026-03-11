; Check that the kernel properties pass updates the initializer of the kernel
; properties global variable. This intentionally does not check that the
; contents of the computed metadata are correct.
;
; RUN: opt --tapir=hip --tapir-hip-arch=gfx906 \
; RUN:     --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:     -passes='loop-spawning,kit-kernel-properties' -S %s\
; RUN:     | FileCheck %s
;
; CHECK: @{{.+}} = private unnamed_addr constant {{.+}} { {{.+}} } #[[KERNEL_PROPS:[0-9]+]]
; CHECK: attributes #[[KERNEL_PROPS]] = {
; CHECK-SAME: "kit_kernel_props"="__kithip_loop_{{.+}}"

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %arrayidx = getelementptr i32, ptr %c, i64 %i
  store i64 %n, ptr %arrayidx, align 4
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
!1 = !{!"tapir.loop.spawn.strategy", i32 3}
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"tapir.loop.lowering.enabled"}

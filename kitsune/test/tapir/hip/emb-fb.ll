; Check that a global variable containing the fat binary is added by the
; hip tapir target.
;
; RUN: opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     -passes='loop-spawning' -S %s \
; RUN:     | FileCheck %s
;
; CHECK: @[[FB:.+]] = constant [{{[0-9]+}} x i8] zeroinitializer
; CHECK-SAME: section ".hip_fatbin"
; CHECK-SAME: align 4096
; CHECK-SAME: #[[ATTR:[0-9]+]]
;
; CHECK: #[[ATTR]] = {
; CHECK-SAME: kit_fb kit_tt(4)

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %arrayidx = getelementptr i64, ptr %c, i64 %i
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

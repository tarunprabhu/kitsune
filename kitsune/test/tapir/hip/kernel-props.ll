; Check that the kernel properties pass updates the initializer of the kernel
; properties global variable.
;
; At the time of writing this test, the only properties that are computed are
; the instruction mix i.e. the counts of various instruction kinds used in the
; kernel. The first two elements of the properties struct are the number of
; memory operations and the number of floating point operations. We only check
; for these since this particular kernel has been crafted such that those values
; can be computed easily. If we change the kernel properties that are computed,
; this test, and the type of the global variable will have to be udpated.
;
; RUN: opt --tapir=hip -passes='loop-spawning,kit-kernel-properties' -S %s \
; RUN:     | FileCheck %s
;
; CHECK: @{{.+}} = private unnamed_addr constant {
; CHECK-SAME: i64, i64, i64, i64 }
; CHECK-SAME: { i64 2, i64 1, {{.+}} }
; CHECK-SAME: !kit.gv.kernel.properties ![[PROPS:[0-9]+]]
;
; CHECK: ![[PROPS]] = !{!"__kithip_loop_{{.+}}"}

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %arrayidx = getelementptr float, ptr %c, i64 %i
  %v = load float, ptr %arrayidx, align 4
  %v2 = fmul float %v, %v
  store float %v2, ptr %arrayidx, align 4
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
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"tapir.loop.lowering.enabled"}
!4 = !{!"tapir.loop.perfect.depth", i32 1}
!5 = !{!"tapir.loop.perfect.level", i32 1}

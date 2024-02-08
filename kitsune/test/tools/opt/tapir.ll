; Check that both the --tapir and --tapir-target are valid options for opt.
;
; RUN: opt --tapir=pthreads -passes="loop-spawning" -disable-output \
; RUN:     -dump-tapir-target-options 2>&1 \
; RUN:     | FileCheck %s
;
; RUN: opt --tapir-target=pthreads -passes="loop-spawning" -disable-output \
; RUN:     -dump-tapir-target-options 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: Primary: pthreads

define void @mset(ptr %a, i64 %n, i64 %v) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %arrayidx = getelementptr i64, ptr %a, i64 %i
  store i64 %v, ptr %arrayidx, align 8
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
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"tapir.loop.target", i32 1}
!3 = !{!"tapir.loop.lowering.enabled"}

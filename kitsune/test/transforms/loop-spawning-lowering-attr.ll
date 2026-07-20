; Loops that do not have the tapir.loop.lowering.enabled attribute should not
; be processed by the loop-spawning pass.
;
; RUN: opt --tapir=pthreads -passes='loop-spawning' -S %s \
; RUN:     | FileCheck %s
;

; CHECK-LABEL: @f
; CHECK: call ptr (i32, ptr, i64, i64, ...) @llvm.kit.async.cpu.threads.launch
define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %arrayidx = getelementptr i64, ptr %c, i64 %i
  store i64 %i, ptr %arrayidx, align 4
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

; CHECK-LABEL: @g
; CHECK-NOT: call ptr (i32, ptr, i64, i64, ...) @llvm.kit.async.cpu.threads.launch
; CHECK: detach within %syncreg2
; CHECK: reattach within %syncreg2
; CHECK: sync within %syncreg2
; CHECK: ret i64 %n
define i64 @g(ptr %c, i64 %n) {
entry:
  %syncreg2 = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg2, label %body, label %latch

body:
  %arrayidx = getelementptr i64, ptr %c, i64 %i
  store i64 %i, ptr %arrayidx, align 4
  reattach within %syncreg2, label %latch

latch:
  %i.next = add i64 %i, 1
  %cmp.i = icmp eq i64 %i.next, %n
  br i1 %cmp.i, label %sync, label %header, !llvm.loop !1

sync:
  sync within %syncreg2, label %exit

exit:
  ret i64 %n
}

!0 = distinct !{!0, !2, !3, !4}
!1 = distinct !{!1, !2, !3}
!2 = !{!"tapir.loop.target", i32 1024}
!3 = !{!"tapir.loop.spawn.strategy", i32 4}
!4 = !{!"tapir.loop.lowering.enabled"}

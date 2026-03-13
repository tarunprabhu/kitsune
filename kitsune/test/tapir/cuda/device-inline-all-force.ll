; Check that the command line option to force inline all device functions
; (including those that have the noinline attribute) is handled correctly.
;
; RUN: opt --tapir=cuda -passes='loop-spawning,emb-prepare' %s \
; RUN:     -emb-inline-all-force \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: define {{.+}} @device_func{{.+}} #[[ATTRS:[0-9]+]]
; CHECK: attributes #[[ATTRS]] = {
; CHECK-SAME: alwaysinline
; CHECK-NOT: noinline

define i64 @device_func(i64 %n) {
  ret i64 %n
}

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %arrayidx = getelementptr i32, ptr %c, i64 %i
  %.call = call i64 @device_func(i64 %n)
  store i64 %.call, ptr %arrayidx, align 4
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

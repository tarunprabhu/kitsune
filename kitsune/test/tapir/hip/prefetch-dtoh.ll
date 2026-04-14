; Check that the prefetch pass inserts device-to-host prefetch calls correctly.
;
; FIXME:
; Currently, we do not insert such prefetch calls, so the checks here ensure
; that this call is not inserted. The test code itself is crafted to ensure that
; the array is accessed on the host after the forall loop, so when we do
; implement device-to-host prefetches, one is likely to be inserted. When we do
; implement this, this comment should be updated/removed.
;
; RUN: opt --tapir=hip -passes='loop-spawning,kit-prefetch' -S %s \
; RUN:     --tapir-gpu-prefetch=true \
; RUN:     | FileCheck %s
;
; CHECK: define {{.+}} @f
; CHECK: %[[STREAM:[0-9]+]] = {{.*}}call ptr @llvm.kit.thread.stream(i32 4)
; CHECK: call {{.+}} @llvm.kit.async.launch.kernel(i32 4
; CHECK-NOT: call {{.+}} @llvm.kit.async.prefetch.dtoh(i32 4
;
; -----------------------------------------------------------------------------

declare void @printf32(float)

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %arrayidx = getelementptr float, ptr %c, i64 %i
  store i64 %i, ptr %arrayidx, align 4
  reattach within %syncreg, label %latch

latch:
  %i.next = add i64 %i, 1
  %cmp.i = icmp eq i64 %i.next, %n
  br i1 %cmp.i, label %sync, label %header, !llvm.loop !0

sync:
  sync within %syncreg, label %end

end:
  %postidx = getelementptr float, ptr %c, i64 %n
  %w = load float, ptr %postidx, align 4
  call void @printf32(float %w)
  br label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"tapir.loop.spawn.strategy", i32 3}
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"tapir.loop.lowering.enabled"}
!4 = !{!"tapir.loop.perfect.depth", i32 1}
!5 = !{!"tapir.loop.perfect.level", i32 1}


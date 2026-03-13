; Check that the prefetch pass inserts host-to-device prefetch calls correctly.
; Currently, we do not compute the number of bytes to be prefetched and always
; pass -1 indicating that the runtime should compute the number of bytes to be
; prefetched. If this changes, this test must be updated.
;
; RUN: opt --tapir=cuda --tapir-gpu-prefetch=true \
; RUN:     -passes='loop-spawning,kit-prefetch' -S %s \
; RUN:     | FileCheck %s
;
; CHECK: define {{.+}} @f1
; CHECK: %[[STREAM:[0-9]+]] = {{.*}}call ptr @llvm.kit.thread.stream(i32 2)
; CHECK-NEXT: call {{.+}} @llvm.kit.async.prefetch.htod(i32 2, ptr %source, i64 -1, ptr %[[STREAM]])
; CHECK-NEXT: call {{.+}} @llvm.kit.async.prefetch.htod(i32 2, ptr %dest, i64 -1, ptr %[[STREAM]])
; CHECK-NEXT: call {{.+}} @llvm.kit.async.launch.kernel(i32 2,
;
; CHECK: define {{.+}} @f2
; CHECK: %[[STREAM:[0-9]+]] = {{.*}}call ptr @llvm.kit.thread.stream(i32 2)
; CHECK-NEXT: call {{.+}} @llvm.kit.async.prefetch.htod(i32 2, ptr %source, i64 -1, ptr %[[STREAM]])
; CHECK-NEXT: call {{.+}} @llvm.kit.async.prefetch.htod(i32 2, ptr %dest, i64 -1, ptr %[[STREAM]])
; CHECK-NEXT: call {{.+}} @llvm.kit.async.launch.kernel(i32 2,
;
; -----------------------------------------------------------------------------

define void @f1(ptr %dest, ptr %source, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %sourceidx = getelementptr float, ptr %source, i64 %i
  %v = load float, ptr %sourceidx, align 4
  %destidx = getelementptr float, ptr %dest, i64 %i
  store float %v, ptr %destidx, align 4
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

define void @f2(ptr %dest, ptr %source, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %sourceidx = getelementptr float, ptr %source, i64 %i
  %v = load float, ptr %sourceidx, align 4
  %destidx = getelementptr float, ptr %dest, i64 %i
  store float %v, ptr %destidx, align 4
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

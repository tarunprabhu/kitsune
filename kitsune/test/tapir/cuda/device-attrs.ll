; Check that the correct attributes have been added to and removed from the
; device function(s).
;
; NOTE: We don't yet fully understand which attributes are actually needed for
; correctness or beneficial for performance. For the immediate future, the
; checks in this test will have to be updated to correctly reflect what CudaABI
; does.
;
; RUN: opt --tapir=cuda --tapir-cuda-arch=sm_72 \
; RUN:     --tapir-cuda-features="+ptx87" \
; RUN:     -passes='loop-spawning,emb-prepare' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: define {{.+}}@device_func{{.+}} #[[ATTRS:[0-9]+]]
; CHECK: attributes #[[ATTRS]] = {
; CHECK-NOT: "personality"
; CHECK-NOT: "tune-cpu"
; CHECK-SAME: kit_device
; CHECK-SAME: "target-cpu"="sm_72"
; CHECK-SAME: "target-features"="+ptx87,sm_72"

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

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.spawn.strategy", i32 3}
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"tapir.loop.lowering.enabled"}

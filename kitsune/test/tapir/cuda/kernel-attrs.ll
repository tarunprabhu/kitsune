; Check that the correct attributes have been added to and removed from the
; kernel function.
;
; NOTE: We don't yet fully understand which attributes are actually needed for
; correctness or beneficial for performance. For the immediate future, the
; checks in this test will have to be updated to correctly reflect what CudaABI
; does.
;
; RUN: opt --tapir=cuda --tapir-cuda-arch=sm_72 --tapir-cuda-features="+ptx87" \
; RUN:     --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:     -passes='loop-spawning' -S %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: define {{.*}}ptx_kernel void {{.+}} #[[ATTRS:[0-9]+]]
; CHECK: attributes #[[ATTRS]] = {
; CHECK-NOT: "personality"
; CHECK-NOT: "tune-cpu"
; CHECK-SAME: kit_kernel
; CHECK-SAME: "target-cpu"="sm_72"
; CHECK-SAME: "target-features"="+ptx87,sm_72"
; CHECK-SAME: "uniform-work-group-size"="true"

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
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"tapir.loop.lowering.enabled"}

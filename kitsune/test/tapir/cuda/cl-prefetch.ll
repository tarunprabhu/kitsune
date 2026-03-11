; Check that the --tapir-gpu-prefetch option is handled correctly
;
; -----------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:     --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:     --tapir-gpu-prefetch=true \
; RUN:     -passes='loop-spawning,kit-prefetch' -S %s \
; RUN:     | FileCheck %s -check-prefix PREFETCH
;
; PREFETCH: define {{.+}} @f
; PREFETCH: call {{.+}} @llvm.kit.async.prefetch.htod(i32 2,
; PREFETCH: call {{.+}} @llvm.kit.async.launch.kernel(i32 2,
; PREFETCH: ret void
; PREFETCH-NEXT: }
;
; -----------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:     --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:     --tapir-gpu-prefetch=false \
; RUN:     -passes='loop-spawning,kit-prefetch' -S %s \
; RUN:     | FileCheck %s -check-prefix NO-PREFETCH
;
; NO-PREFETCH: define {{.+}} @f
; NO-PREFETCH-NOT: call {{.+}} @llvm.kit.async.prefetch.htod(i32 2,
; NO-PREFETCH: call {{.+}} @llvm.kit.async.launch.kernel(i32 2,
; NO-PREFETCH: ret void
; NO-PREFETCH-NEXT: }
;
; -----------------------------------------------------------------------------

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
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"tapir.loop.lowering.enabled"}

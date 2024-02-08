; ------------------------------------------------------------------------------
; If no explicit optimization level is specified, the optimization level set by
; the "frontend" (in this case, the value passed to the tapir-lower meta-pass)
; should be used.
;
; ------------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:     --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:     -passes='tapir-lowering<O1>,emb-optimize' -disable-output %s \
; RUN:     -emb-print-pipeline-passes \
; RUN:     | FileCheck %s --check-prefix=O1
;
; O1: NVVMReflectPass
; O1-SAME: LoopUnrollPass<O1>
;
; ------------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:     --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:     -passes='tapir-lowering<O2>,emb-optimize' -disable-output %s \
; RUN:     -emb-print-pipeline-passes \
; RUN:     | FileCheck %s --check-prefix=O2
;
; O2: NVVMReflectPass
; O2-SAME: LoopUnrollPass<O2>
;
; ------------------------------------------------------------------------------

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

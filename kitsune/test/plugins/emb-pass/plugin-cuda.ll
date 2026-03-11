; REQUIRES: kitsune-examples, kitsune-cuda
;
; Check that the embedded module passes in a pass plugin is registered with the
; Kitsune pipeline when the kit-lowering meta-pass is used.
;
; NOTE: We only check for the presence of external_func called in the forall
; loop. A second function, the kernel function, will also be present but we
; do not check for it because the name of that function is not guaranteed to
; be consistent.
;
; NOTE: We have separate tests for all tapir targets that use embedded bitcode
; to ensure that, if at least one is built, that this functionality is tested.
; They cannot all be compiled into a single test because that would be treated
; as multi-target compilation, which is not yet supported.
;
; RUN: opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:     --tapir-cuda-runtime-bc=%S/input/libdevice.bc \
; RUN:     --load-pass-plugin=%kit-emb-pass-plugin-demo \
; RUN:     -passes='kit-lowering<O1>' -disable-output %s \
; RUN:     | FileCheck %s
;
; CHECK-DAG: external_func
; CHECK-DAG: llvm.nvvm.read.ptx.sreg.tid.x

target triple = "x86_64-unknown-linux-gnu"

declare i64 @external_func(i64)

define void @mset(ptr %a, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %v = call i64 @external_func(i64 %i)
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
!1 = !{!"tapir.loop.spawn.strategy", i32 3}
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"tapir.loop.lowering.enabled"}

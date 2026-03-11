; Check that linking libdevice bitcode works as expected.
;
; RUN: opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:     --tapir-cuda-runtime-bc=%S/input/libdevice.ll %s \
; RUN:     -passes='loop-spawning,emb-resolve-libdevice-calls,emb-link-libdevice-bitcode' \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; We should only link in what is actually needed
; CHECK-NOT: define {{.+}} @__nv_fast_sinf
; CHECK-NOT: define {{.+}} @__nv_cosf
;
; CHECK-LABEL: define {{.+}} @__kitcu_
; CHECK: tail call float @__nv_sinf
; CHECK: tail call fast float @__nv_fast_cosf
; CHECK-DAG: define {{.+}} @__nv_sinf
; CHECK-DAG: define {{.+}} @__nv_fast_cosf

declare float @sinf(float)
declare float @cosf(float)

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %arrayidx = getelementptr i32, ptr %c, i64 %i
  %asf = sitofp i64 %n to float
  %sin = tail call float @sinf(float %asf)
  %cos = tail call fast float @cosf(float %sin)
  %asi = fptosi float %cos to i64
  store i64 %asi, ptr %arrayidx, align 4
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

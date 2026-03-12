; Check that functions with the __nv and __nv_fast prefixes are left as is.
;
; RUN: opt --tapir=cuda --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:     -passes='loop-spawning,emb-resolve-libdevice-calls' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: %[[ASF:.+]] = sitofp i64
; CHECK-NEXT: %[[SIN:.+]] = tail call float @__nv_sinf(float %[[ASF]])
; CHECK-NEXT: %[[SINF:.+]] = tail call float @__nv_fast_sinf(float %[[SIN]])
; CHECK-NEXT: store float %[[SINF]],

declare float @__nv_sinf(float)
declare float @__nv_fast_sinf(float)

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %arrayidx = getelementptr float, ptr %c, i64 %i
  %asf = sitofp i64 %i to float
  %sin = tail call float @__nv_sinf(float %asf)
  %sinf = tail call float @__nv_fast_sinf(float %sin)
  store float %sinf, ptr %arrayidx, align 4
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

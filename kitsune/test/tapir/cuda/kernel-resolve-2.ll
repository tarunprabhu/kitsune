; Check that the same function encountered more than once is handled correctly
;
; RUN: opt --tapir=cuda --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:     -passes='loop-spawning,emb-resolve-libdevice-calls' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: %[[ASF:.+]] = sitofp i64
; CHECK-NEXT: %[[SIN1:.+]] = tail call float @__nv_sinf(float %[[ASF]])
; CHECK-NEXT: %[[SIN2:.+]] = tail call float @__nv_sinf(float %[[SIN1]])
; CHECK-NEXT: store float %[[SIN2]],
; CHECK-COUNT-1: declare float @__nv_sinf(float)
; CHECK-NOT: declare float @__nv_sinf{{.+}}(float)

declare float @sinf(float)

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
  %sin1 = tail call float @sinf(float %asf)
  %sin2 = tail call float @sinf(float %sin1)
  store float %sin2, ptr %arrayidx, align 4
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

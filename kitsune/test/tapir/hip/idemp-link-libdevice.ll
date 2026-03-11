; Check that linking libdevice bitcode multiple times does not result in
; duplicate definitions or creation of a symbol with an additional LLVM suffix.
;
; RUN: opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:     --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:     -passes='loop-spawning,emb-resolve-libdevice-calls,emb-link-libdevice-bitcode,emb-link-libdevice-bitcode' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-COUNT-1: @__ocmltbl_M32_J0{{.*}} =
; CHECK-COUNT-1: define {{.+}} @__ocml_j0_f32

declare float @j0f(float)

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
  %.j0 = tail call float @j0f(float %asf)
  store float %.j0, ptr %arrayidx, align 4
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
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"tapir.loop.lowering.enabled"}

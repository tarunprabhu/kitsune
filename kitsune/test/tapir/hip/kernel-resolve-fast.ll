; Check that device functions are resolved correctly. This is a very basic test.
; We really should do something a bit more comprehensive
;
; RUN: opt --tapir=hip --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:     -passes='loop-spawning,emb-resolve-libdevice-calls' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; Kitsune does not yet support fast math functions on AMDGPU
; XFAIL: *
; CHECK: tail call fast float @__ocml_fast_sinf
; CHECK: tail call fast float @__ocml_fast_cosf

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %arrayidx = getelementptr float, ptr %c, i64 %i
  %asf = sitofp i64 %n to float
  %sin = tail call fast float @sinf(float %asf)
  %cos = tail call fast float @cosf(float %sin)
  store float %cos, ptr %arrayidx, align 4
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

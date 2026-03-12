; Check that llvm intrinsics are resolved correctly. This only checks for some
; intrinsics which is not terribly useful in the grand scheme of things.
;
; RUN: opt --tapir=cuda --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:     -passes='loop-spawning,emb-resolve-libdevice-calls' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: tail call float @__nv_tanf
; CHECK: tail call double @__nv_sqrt

declare float @llvm.tan.f32(float)
declare double @llvm.sqrt.f64(double)

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
  %.sc32 = tail call float @llvm.tan.f32(float %asf)
  %.cst = fpext float %.sc32 to double
  %.sc64 = tail call double @llvm.sqrt.f64(double %.cst)
  store double %.sc64, ptr %arrayidx, align 4
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

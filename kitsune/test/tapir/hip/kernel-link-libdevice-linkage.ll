; When resolving libdevice functions in the emb-resolve-libdevice-calls pass, a
; declaration of the libdevice function is added into the device module. At this
; time, the linkage of the function is changed to be external. When the
; definitions of the functions are provided in the emb-link-libdevice-bitcode pass,
; these linkages should be overridden with those in the libdevice module.
;
; RUN: opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:     --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:     -passes='loop-spawning,emb-resolve-libdevice-calls,emb-link-libdevice-bitcode' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-DAG: define linkonce_odr protected {{.*}}float @__ocml_j0_f32
; CHECK-DAG: define linkonce_odr hidden {{.*}}double @__ocml_sqrt_f64

declare float @j0f(float)
declare double @sqrt(double)

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
  %.acos = tail call float @j0f(float %asf)
  %.cst = fpext float %.acos to double
  %.sqrt = tail call double @sqrt(double %.cst)
  store double %.sqrt, ptr %arrayidx, align 4
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

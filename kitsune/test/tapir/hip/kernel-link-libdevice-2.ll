; Check that linking libdevice bitcode works as expected. The functions used
; here are defined in two separate files.
;
; RUN: opt --tapir=hip %s \
; RUN:     --tapir-hip-runtime-bcs=%S/input/libdevice.ll,%S/input/libdevice-2.ll \
; RUN:     -passes='loop-spawning,emb-resolve-libdevice-calls,emb-link-libdevice-bitcode'\
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; We should only link in what is actually needed
; CHECK-NOT: define {{.+}} @__ocml_acos_f64
; CHECK-NOT: define {{.+}} @__ocml_sqrt_f32
; CHECK-NOT: define {{.+}} @__ocml_erfc_f32
;
; CHECK-LABEL: define {{.+}} @__kithip_
; CHECK: tail call float @__ocml_acos_f32
; CHECK: tail call double @__ocml_erfc_f64
; CHECK-DAG: define {{.+}} @__ocml_acos_f32
; CHECK-DAG: define {{.+}} @__ocml_erfc_f64

declare float @acosf(float)
declare double @erfc(double)

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
  %.acos = tail call float @acosf(float %asf)
  %.cst = fpext float %.acos to double
  %.erfc = tail call double @erfc(double %.cst)
  store double %.erfc, ptr %arrayidx, align 4
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

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"tapir.loop.spawn.strategy", i32 3}
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"tapir.loop.lowering.enabled"}
!4 = !{!"tapir.loop.perfect.depth", i32 1}
!5 = !{!"tapir.loop.perfect.level", i32 1}

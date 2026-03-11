; Check that the same function encountered more than once is handled correctly
;
; RUN: opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:     --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:     -passes='loop-spawning,emb-resolve-libdevice-calls' -S %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: %[[ASF:.+]] = sitofp i64
; CHECK-NEXT: %[[SIN1:.+]] = tail call double @__ocml_acos_f64(double %[[ASF]])
; CHECK-NEXT: %[[SIN2:.+]] = tail call double @__ocml_acos_f64(double %[[SIN1]])
; CHECK-NEXT: store double %[[SIN2]],
; CHECK-COUNT-1: declare {{.*}}double @__ocml_acos_f64(double{{.*}})
; CHECK-NOT: declare {{.*}}double @__ocml_acos_f64{{.+}}(double{{.*}})

declare double @acos(double) #2

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %arrayidx = getelementptr float, ptr %c, i64 %i
  %asf = sitofp i64 %i to double
  %sin1 = tail call double @acos(double %asf)
  %sin2 = tail call double @acos(double %sin1)
  store double %sin2, ptr %arrayidx, align 4
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

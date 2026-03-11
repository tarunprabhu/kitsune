; Check that any constant global variables are handled correctly. They should
; not be copied memcpy'ed, and they should not be registered with the runtime.
;
; RUN: opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     -passes='loop-spawning,kit-ctors' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @[[FB:.+]] = constant {{.+}} #[[ATTR:[0-9]+]]
;
; CHECK: define {{.+}} @f
; CHECK-NOT: llvm.kit.symbol.device.ptr
; CHECK-NOT: llvm.kit.symbol.memcpy.htod
; CHECK: %[[TS:.+]] = {{.*}}call {{.+}} @llvm.kit.async.launch.kernel(i32 4, ptr @[[FB]],
; CHECK-NOT: llvm.kit.symbol.memcpy.dtoh
;
; CHECK: define {{.+}} @.kithip.ctor{{[^(]*}}
; CHECK: call {{.+}} @__hipRegisterFatBinary
; CHECK-NOT: call {{.+}} @__hipRegisterVar
;
; CHECK: #[[ATTR]] = { kit_fb kit_tt(4) }

target triple = "x86_64-pc-linux-gnu"

@v137 = external constant i32, align 4

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %0 = load i32, ptr @v137, align 4
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %arrayidx = getelementptr i32, ptr %c, i64 %i
  store i32 %0, ptr %arrayidx, align 4
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

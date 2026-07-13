; Check that any constant global variables are handled correctly. They should
; not be copied memcpy'ed, and they should not be registered with the runtime.
;
; RUN: opt --tapir=cuda -passes='loop-spawning,kit-ctors' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @[[FB:.+]] = constant {{.+}} !kit.gv ![[MD:[0-9]+]]
;
; CHECK: define {{.+}} @f
; CHECK-NOT: llvm.kit.gpu.symbol.address
; CHECK-NOT: llvm.kit.gpu.symbol.memcpy.htod
; CHECK: %[[TS:.+]] = {{.*}}call {{.+}} @llvm.kit.async.gpu.kernel.launch(i32 2, ptr @[[FB]],
; CHECK-NOT: llvm.kit.gpu.symbol.memcpy.dtoh
;
; CHECK: define {{.+}} @.kit.cuda.ctor{{[^(]*}}
; CHECK: call {{.+}} @llvm.kit.gpu.register.devcode
; CHECK-NOT: call {{.+}} @llvm.kit.gpu.register.global
; CHECK: call {{.+}} @llvm.kit.gpu.register.devcode.end
;
; CHECK-DAG: ![[MD]] = distinct !{![[MD]], ![[DC:[0-9]+]]}
; CHECK-DAG: ![[DC]] = !{!"kit.gv.device.code", i32 2}

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

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"tapir.loop.spawn.strategy", i32 3}
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"tapir.loop.lowering.enabled"}
!4 = !{!"tapir.loop.perfect.depth", i32 1}
!5 = !{!"tapir.loop.perfect.level", i32 1}

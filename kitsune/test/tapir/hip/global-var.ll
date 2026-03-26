; Check that any non-const global variables are handled correctly when
; generating kernel launch calls. They should be memcpy'ed to and from the
; before and after launch calls and must be registered with the runtime in
; the ctor for kitsune's runtime.
;
; RUN: opt --tapir=hip -passes='loop-spawning,kit-ctors' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @[[FB:.+]] = constant {{.+}} !kit.gv ![[MD:[0-9]+]]
; CHECK-DAG: @[[HOSTVAR:.+]] = external {{.+}} i32
; CHECK-DAG: @[[VARNAME:.+]] = private unnamed_addr constant [5 x i8] c"v137\00"
;
; CHECK: define {{.+}} @f
; CHECK: %[[PTR1:.+]] = {{.*}}call {{.+}} @llvm.kit.symbol.device.ptr(i32 4, ptr @[[FB]], ptr @[[VARNAME]])
; CHECK: call {{.+}} @llvm.kit.symbol.memcpy.htod(i32 4, ptr %[[PTR1]], ptr @[[HOSTVAR]], i64 4)
; CHECK: %[[TS:.+]] = {{.*}}call {{.+}} @llvm.kit.async.launch.kernel(i32 4, ptr @[[FB]],
; CHECK: %[[PTR2:.+]] = {{.*}}call {{.+}} @llvm.kit.symbol.device.ptr(i32 4, ptr @[[FB]], ptr @[[VARNAME]])
; CHECK: call {{.+}} @llvm.kit.symbol.memcpy.dtoh(i32 4, ptr @[[HOSTVAR]], ptr %[[PTR2]], i64 4)
; CHECK: ret void
; CHECK-NEXT: }
;
; CHECK: define {{.+}} @.kithip.ctor{{[^(]*}}
; CHECK: %[[HANDLE:.+]] = call ptr @__hipRegisterFatBinary
; CHECK: call {{.+}} @__hipRegisterVar(ptr %[[HANDLE]], ptr @[[HOSTVAR]], ptr @[[VARNAME]]
;
; CHECK-DAG: ![[MD]] = distinct !{![[MD]], ![[DC:[0-9]+]]}
; CHECK-DAG: ![[DC]] = !{!"kit.gv.device.code", i32 4}

target triple = "x86_64-pc-linux-gnu"

@v137 = external global i32, align 4

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %0 = load i32, ptr @v137, align 4
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
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"tapir.loop.lowering.enabled"}
!4 = !{!"tapir.loop.perfect.depth", i32 1}
!5 = !{!"tapir.loop.perfect.level", i32 1}

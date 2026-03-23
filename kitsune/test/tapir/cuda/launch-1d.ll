; Check that a launch call and a fat binary are present in the host and that
; the arguments to the launch call are as expected when lowering a tapir loop
; nest of depth 1.
;
; RUN: opt --tapir=cuda -passes='loop-spawning' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @[[FB:.+]] = constant {{.+}} !kit.gv.device.code ![[TT:[0-9]+]]
; CHECK-DAG: @[[G_KNAME:.+]] = private unnamed_addr constant [{{[0-9]+}} x i8] c"[[KNAME:.+]]\00"
; CHECK-DAG: @[[G_KERNEL_PROPS:.+]] = private unnamed_addr constant {{.+}} zeroinitializer, !kit.gv.kernel.properties ![[KP:[0-9]+]]
;
; CHECK: define {{.+}} @f(ptr {{.*}}%[[C:.+]], i64 {{.*}}%[[N:.+]])
;
; CHECK: %[[STREAM:.+]] = {{.*}}call {{.+}} @llvm.kit.thread.stream(i32 2)
; CHECK-NOT: {{.*}}call {{.+}} @llvm.kit.async.prefetch.htod
;
; CHECK: %{{[0-9]+}} = {{.*}}call {{.+}} @llvm.kit.async.launch.kernel(
; CHECK-SAME: i32 2,
; CHECK-SAME: ptr {{.*}}@[[FB]],
; CHECK-SAME: ptr {{.*}}@[[G_KNAME]],
; CHECK-SAME: i64 0,
; CHECK-SAME: i64 0,
; CHECK-SAME: i64 %[[N]],
; CHECK-SAME: i32 0,
; CHECK-SAME: ptr {{.*}}@[[G_KERNEL_PROPS]],
; CHECK-SAME: ptr %[[STREAM]],
; CHECK-SAME: i64 0,
; CHECK-SAME: i64 %[[N]],
; CHECK-SAME: ptr %[[C]],
; CHECK-SAME: i64 %[[N]])
;
; By default, we always enter a sync immediately after the launch. A later
; optimization pass may (re)move this if appropriate
;
; CHECK: call {{.+}} @llvm.kit.sync.stream(i32 2, ptr %[[STREAM]])
; CHECK: ret void
; CHECK-NEXT: }
;
; CHECK-DAG: ![[TT]] = !{i32 2}
; CHECK-DAG: ![[KP]] = !{i32 2, !"[[KNAME]]"}

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %arrayidx = getelementptr i32, ptr %c, i64 %i
  store i64 %n, ptr %arrayidx, align 4
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

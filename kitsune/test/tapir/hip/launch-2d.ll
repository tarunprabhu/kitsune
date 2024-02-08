; Check that a launch call and a fat binary are present in the host and that
; the arguments to the launch call are as expected when lowering a tapir loop
; nest of depth 2.
;
; RUN: opt --tapir=hip -passes='loop-spawning' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @[[FB:.+]] = constant {{.+}} !kit.gv ![[MDDC:[0-9]+]]
; CHECK-DAG: @[[G_KNAME:.+]] = private unnamed_addr constant [{{[0-9]+}} x i8] c"[[KNAME:.+]]\00"
; CHECK-DAG: @[[G_KERNEL_PROPS:.+]] = private unnamed_addr constant {{.+}} zeroinitializer, !kit.gv ![[MDKP:[0-9]+]]
;
; CHECK-LABEL: define {{.+}}@pp
; CHECK-SAME: i64 {{[^%]*}}%[[M:[^,]+]],
; CHECK-SAME: i64 {{[^%]*}}%[[N:[^)]+]],
; CHECK-SAME: ptr {{[^%]*}}%[[C:[^,]+]])
;
; CHECK: %[[STREAM:.+]] = {{.*}}call {{.+}} @llvm.kit.gpu.stream.new(i32 4)
; CHECK-NOT: {{.*}}call {{.+}} @llvm.kit.async.gpu.prefetch.htod
;
; CHECK: %{{[0-9]+}} = {{.*}}call {{.+}} @llvm.kit.async.gpu.kernel.launch(
; CHECK-SAME: i32 4,
; CHECK-SAME: ptr {{.*}}@[[FB]],
; CHECK-SAME: ptr {{.*}}@[[G_KNAME]],
; CHECK-SAME: i64 0,
; CHECK-SAME: i64 %[[M]],
; CHECK-SAME: i64 %[[N]],
; CHECK-SAME: i32 0,
; CHECK-SAME: ptr {{.*}}@[[G_KERNEL_PROPS]],
; CHECK-SAME: ptr %[[STREAM]],
; CHECK-SAME: i64 0,
; CHECK-SAME: i64 %[[M]],
; CHECK-SAME: i64 0,
; CHECK-SAME: i64 %[[N]],
; CHECK-SAME: ptr %[[C]],
; CHECK-SAME: i64 %[[N]])
;
; By default, we always enter a sync immediately after the launch. A later
; optimization pass may (re)move this if appropriate
;
; CHECK: call {{.+}} @llvm.kit.gpu.stream.sync(i32 4, ptr %[[STREAM]])
; CHECK: ret void
; CHECK-NEXT: }
;
; CHECK-DAG: ![[MDDC]] = distinct !{![[MDDC]], ![[DC:[0-9]+]]}
; CHECK-DAG: ![[DC]] = !{!"kit.gv.device.code", i32 4}
; CHECK-DAG: ![[MDKP]] = distinct !{![[MDKP]], ![[KP:[0-9]+]]}
; CHECK-DAG: ![[KP]] = !{!"kit.gv.kernel.properties", i32 4, !"[[KNAME]]"}

define void @pp(i64 %m, i64 %n, ptr %c) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.j = tail call token @llvm.syncregion.start()
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  call void @ext2(ptr %c, i64 %i, i64 %j)
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !1

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

declare void @ext2(ptr, i64, i64)

!0 = distinct !{!0, !2, !3, !4, !5, !6}
!1 = distinct !{!1, !2, !3, !7}
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"tapir.loop.spawn.strategy", i32 3}
!4 = !{!"tapir.loop.lowering.enabled"}
!5 = !{!"tapir.loop.perfect.depth", i32 2}
!6 = !{!"tapir.loop.perfect.level", i32 1}
!7 = !{!"tapir.loop.perfect.level", i32 2}

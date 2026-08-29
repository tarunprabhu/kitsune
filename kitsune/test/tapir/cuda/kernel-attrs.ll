; Check that the correct attributes have been added to and removed from the
; kernel function.
;
; NOTE: We don't yet fully understand which attributes are actually needed for
; correctness or beneficial for performance. As a result the attributes on the
; kernel may change. This test should be kept up to date with the changes made
; in the cuda tapir target.
;
; RUN: opt --tapir=cuda --tapir-cuda-arch=sm_72 \
; RUN:     --tapir-cuda-features="+ptx87" \
; RUN:     -passes='loop-spawning' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: define {{.*}}ptx_kernel void {{[^#]+}}
; CHECK-SAME: #[[ATTRS:[0-9]+]]
; CHECK-SAME: !kit.func ![[MD:[0-9]+]]
;
; CHECK: attributes #[[ATTRS]] = {
; CHECK-NOT: "personality"
; CHECK-NOT: "tune-cpu"
; CHECK-SAME: "target-cpu"="sm_72"
; CHECK-SAME: "target-features"="+ptx87,sm_72"
; CHECK-SAME: "uniform-work-group-size"="true"
;
; CHECK-DAG: ![[MD]] = distinct !{![[MD]], ![[KERNEL:[0-9+]]]}
; CHECK-DAG: ![[KERNEL]] = !{!"kit.func.kernel", i32 2}

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
  %arrayidx = getelementptr i64, ptr %c, i64 %j
  %product = mul i64 %m, %n
  store i64 %product, ptr %arrayidx
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

!0 = distinct !{!0, !2, !3, !4, !5, !6}
!1 = distinct !{!1, !2, !3, !7}
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"tapir.loop.spawn.strategy", i32 3}
!4 = !{!"tapir.loop.lowering.enabled"}
!5 = !{!"tapir.loop.perfect.depth", i32 2}
!6 = !{!"tapir.loop.perfect.level", i32 1}
!7 = !{!"tapir.loop.perfect.level", i32 2}

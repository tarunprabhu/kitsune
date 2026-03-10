; The grainsize is always set to 1 currently. Check that this is the case in
; the kernel immediately after loop spawning.
;
; RUN: opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     -passes='tapir-lowering<O1>' -S %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: define {{.*}}amdgpu_kernel
; CHECK: [[ENTRY:[^:]+]]:
; CHECK: %[[IV_START:.+]] = add {{.*}}i64 %{{.+}}, %{{.+}}
; CHECK: %[[IV_END:.+]] = add {{.*}}i64 %[[IV_START]], 1
; CHECK: %[[IV:.+]] = phi i64
; CHECK-SAME: %[[IV_START]], %[[ENTRY]]
; CHECK: %[[INC:.+]] = add {{.*}}i64 %[[IV]], 1
; CHECK: icmp eq i64 %[[INC]], %[[IV_END]]

define void @p(ptr %a, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
  %a.i = getelementptr inbounds i64, ptr %a, i64 %i
  store i64 %i, ptr %a.i
  reattach within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %exitcond.i.not = icmp eq i64 %inc.i, %n
  br i1 %exitcond.i.not, label %for.i.sync, label %for.i.header, !llvm.loop !0

for.i.sync:
  sync within %syncreg, label %for.i.exit

for.i.exit:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.target", i32 4}
!2 = !{!"tapir.loop.spawn.strategy", i32 3}

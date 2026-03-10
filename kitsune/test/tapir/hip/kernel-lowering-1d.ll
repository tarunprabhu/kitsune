; Check that the kernel generated immediately after lowering of a tapir loop
; nest of depth 1 is as expected. In this case, a single loop will be present
; in the kernel body. The start index of the loop will be an index computed
; from cuda's intrinsics that return threadIdx.x and blockIdx.x. The end will
; depend on the grainsize, but we test for the expected grainsize used in a
; different test.
;
; RUN: opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     -passes='tapir-lowering<O2>' -S %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: define {{.*}}amdgpu_kernel
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK: %[[IV_START:.+]] = add {{.*}}i64 %{{.+}}, %{{.+}}
; CHECK: %[[IV_END:.+]] = add {{.*}}i64 %[[IV_START]]
; CHECK: [[HEADER:.+]]:
; CHECK-NEXT: %[[IV:.+]] = phi i64
; CHECK-SAME: %[[IV_START]], %[[ENTRY]]
; CHECK-NEXT: br label %[[BODY:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: getelementptr
; CHECK-NEXT: store
; CHECK-NEXT: br label %[[LATCH:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH]]:
; CHECK-NEXT: %[[INC:.+]] = add {{.*}}i64 %[[IV]], 1
; CHECK-NEXT: %[[COND:.+]] = icmp eq i64 %[[INC]], %[[IV_END]]
; CHECK-NEXT: br i1 %[[COND]], label %[[EXIT:.+]], label %[[HEADER]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void

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


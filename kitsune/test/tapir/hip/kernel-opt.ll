; Check that when the standard sequence of optimization passes are run on the
; code, the results as expected.
;
; ------------------------------------------------------------------------------
;
; We have to set the optimization level of tapir lowering to non-zero for it to
; work. However, we can override the optimization level used on the embedded
; bitcode module. The lowering only adjusts the bounds of the original tapir
; loop. Setting the optimization level to O0 retains this loop.
;
; RUN: opt --tapir=hip -passes='loop-spawning,emb-optimize' %s \
; RUN:     -emb-O0 \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s --check-prefix=O0
;
; O0: define {{.+}} @__kithip_{{.+}}(i64
; O0: = phi i64
;
; ------------------------------------------------------------------------------
;
; NOTE: This assumes that the grainsize is 1. For now, this is hard-coded into
; the hip tapir target. If we ever allow this to be configurable, and set the
; default value to something other than 1, this needs to be changed.
;
; If compiling with optimizations, the loop will be removed since the trip
; is determined to be 1. If the grain size is changed to be greater than 1, we
; may need to check for unrolling.
;
; RUN: opt --tapir=hip -passes='loop-spawning,emb-optimize' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s --check-prefix=O2
;
; O2-NOT: = phi i64
; O2: define {{.+}} @__kithip_{{[^(]+}}(
; O2-SAME: i64 {{[^%]*}}%[[LB:[^,]+]],
; O2-SAME: i64 {{[^%]*}}%[[TC:[^,]+]],
; O2-SAME: ptr {{[^%]*}}%[[BUF:[^,]+]],
; O2-SAME: i64 {{[^%]*}}%[[N:[^)]+]])
; O2-SAME: {{.*}}#[[ATTRS:[0-9]+]]
; O2-NEXT: [[BBENTRY:.+]]:
; O2-NEXT: %[[BUFCST:.+]] = addrspacecast ptr %[[BUF]] to ptr addrspace(1)
; O2-NEXT: %[[TIDX:.+]] = {{.*}}call i32 @llvm.kit.gpu.thread.id.x()
; O2-NEXT: %[[BIDX:.+]] = {{.*}}call i32 @llvm.kit.gpu.block.id.x()
; O2-NEXT: %[[BDIM:.+]] = {{.*}}call i32 @llvm.kit.gpu.block.size.x()
; O2-NEXT: %[[BOFF:.+]] = mul i32 %[[BDIM]], %[[BIDX]]
; O2-NEXT: %[[IVBEG32:.+]] = add i32 %[[BOFF]], %[[TIDX]]
; O2-NEXT: %[[IVBEG:.+]] = zext i32 %[[IVBEG32]] to i64
; O2-NEXT: %[[IVCOND:.+]] = icmp ugt i64 %[[TC]], %[[IVBEG]]
; O2-NEXT: br i1 %[[IVCOND]], label %[[BBBODY:[^,]+]], label %[[BBEXIT:.+]]
; O2: [[BBBODY]]:
; O2-NEXT: %[[ARRIDX:.+]] = getelementptr {{.+}}, ptr {{.*}}%[[BUFCST]], i64 %[[IVBEG]]
; O2-NEXT: store i64 %[[N]], ptr {{.*}}%[[ARRIDX]]
; O2-NEXT: br label %[[BBEXIT]]
; O2: [[BBEXIT]]:
; O2-NEXT: ret void
;
; O2: attributes #[[ATTRS]] = {
; O2-SAME: kit_kernel
;
; ------------------------------------------------------------------------------

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
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"tapir.loop.lowering.enabled"}
!4 = !{!"tapir.loop.perfect.depth", i32 1}
!5 = !{!"tapir.loop.perfect.level", i32 1}

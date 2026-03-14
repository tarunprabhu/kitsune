; Check that the induction variable for a 1D launch is computed correctly and
; bypasses the body of the loop if out of bounds
;
; RUN: opt --tapir=cuda -passes='loop-spawning' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: define {{[^(]+}}(
; CHECK-SAME: i64 {{[^%]*}}%[[LB:[^,]+]],
; CHECK-SAME: i64 {{[^%]*}}%[[TC:[^,]+]],
; CHECK-SAME: i64 {{[^)]+}})
; CHECK-SAME: #[[ATTRS:[0-9]+]]
; CHECK: %[[TIDX:.+]] = {{.*}}call i32 @llvm.kit.gpu.thread.id.x()
; CHECK: %[[BIDX:.+]] = {{.*}}call i32 @llvm.kit.gpu.block.id.x()
; CHECK: %[[BDIM:.+]] = {{.*}}call i32 @llvm.kit.gpu.block.size.x()
; CHECK: %[[BOFF:.+]] = mul i32 %[[BIDX]], %[[BDIM]]
; CHECK: %[[IVBEG32:.+]] = add i32 %[[TIDX]], %[[BOFF]]
; CHECK: %[[IVBEG:.+]] = zext i32 %[[IVBEG32]] to i64
; CHECK: %[[IVCOND:.+]] = icmp uge i64 %[[IVBEG]], %[[TC]]
; CHECK-NEXT: br i1 %[[IVCOND]], label %[[BBEXIT:[^,]+]], label %[[BBHEADER:.+]]
; CHECK: [[BBHEADER]]:
; CHECK-NEXT: phi i64
; CHECK: [[BBEXIT]]:
; CHECK-NEXT: ret void
;
; CHECK: attributes #[[ATTRS]] = {
; CHECK-SAME: kit_kernel

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

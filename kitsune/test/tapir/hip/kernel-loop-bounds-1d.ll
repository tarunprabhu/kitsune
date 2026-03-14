; Check that the loop bounds of the tapir loop are correctly replaced in the
; outlined kernel function when a 1D kernel is launched.
;
; NOTE: The upper bounds is determined by the grainsize. We deliberately do not
; check for the actual grainsize here. That will be tested elsewhere. This is
; only intended to check the bounds.
;
; RUN: opt --tapir=hip -passes='loop-spawning' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: define {{.+}}(
; CHECK-SAME: i64 {{[^,]+}},
; CHECK-SAME: i64 {{[^,]+}},
; CHECK-SAME: ptr {{.*}}%[[BUF:[^,]+]],
; CHECK-SAME: i64 {{.*}}%[[N:[^)]+]])
; CHECK-NEXT: [[PREHEADER:.+]]:
; CHECK: %[[TIDX:.+]] = {{.*}}call i32 @llvm.kit.gpu.thread.id.x()
; CHECK: %[[BDIM:.+]] = {{.*}}call i32 @llvm.kit.gpu.block.size.x()
; CHECK: %[[BIDX:.+]] = {{.*}}call i32 @llvm.kit.gpu.block.id.x()
; CHECK: %[[BOFF:.+]] = mul i32 %[[BIDX]], %[[BDIM]]
; CHECK: %[[IVBEG32:.+]] = add i32 %[[TIDX]], %[[BOFF]]
; CHECK: %[[IVBEG:.+]] = zext i32 %[[IVBEG32]] to i64
; CHECK: %[[IVEND:.+]] = add i64 %[[IVBEG]]
;
; CHECK: [[HEADER:.+]]:
; CHECK: %[[IV:.+]] = phi i64
; CHECK-SAME: [ %[[IVBEG]], %[[PREHEADER]] ]
; CHECK-SAME: [ %[[IVNEXT:.+]], %[[LATCH:.+]] ]
; CHECK: %[[IVNEXT:.+]] = add {{.*}}i64 %[[IV]]
; CHECK: %[[COND:.+]] = icmp eq i64 %[[IVNEXT]], %[[IVEND]]
; CHECK: br i1 %[[COND]], label %[[EXIT:.+]], label %[[HEADER]]
;
; CHECK: [[EXIT]]:
; CHECK-NEXT: ret void

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

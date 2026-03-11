; Check that the loop bounds of the tapir loop are correctly replaced in the
; outlined kernel function when a 1D kernel is launched.
;
; NOTE: Currently, we use a grainsize of 1 i.e. every thread computes a single
; iteration of the tapir loop, but if that changes, this test must be updated.
;
; RUN: opt --tapir=hip --tapir-hip-arch="gfx90a" \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     -passes='loop-spawning' -S %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: define {{.+}}(
; CHECK-SAME: i64 {{[^,]+}},
; CHECK-SAME: i64 {{[^,]+}},
; CHECK-SAME: i64 {{[^,]+}},
; CHECK-SAME: ptr {{.*}}%[[BUF:[^,]+]],
; CHECK-SAME: i64 {{.*}}%[[N:[^)]+]])
; CHECK-NEXT: [[PREHEADER:.+]]:
; CHECK: %[[WITEM:.+]] = {{.*}}call i32 @llvm.amdgcn.workitem.id.x()
; CHECK: %[[TID:.+]] = zext i32 %[[WITEM]] to i64
; CHECK: %[[BDIM:.+]] = {{.*}}call i64 @__ockl_get_local_size(i32 0)
; CHECK: %[[WGRP:.+]] = {{.*}}call i32 @llvm.amdgcn.workgroup.id.x()
; CHECK: %[[BIDX:.+]] = zext i32 %[[WGRP]] to i64
; CHECK: %[[BOFF:.+]] = mul i64 %[[BIDX]], %[[BDIM]]
; CHECK: %[[IV_START:.+]] = add i64 %[[TID]], %[[BOFF]]
; CHECK: %[[IV_END:.+]] = add i64 %[[IV_START]], 1
;
; CHECK: [[HEADER:.+]]:
; CHECK: %[[IV:.+]] = phi i64
; CHECK-SAME: [ %[[IV_START]], %[[PREHEADER]] ]
; CHECK-SAME: [ %[[IV_NEXT:.+]], %[[LATCH:.+]] ]
; CHECK: %[[IV_NEXT:.+]] = add {{.*}}i64 %[[IV]], 1
; CHECK: %[[COND:.+]] = icmp eq i64 %[[IV_NEXT]], %[[IV_END]]
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

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.spawn.strategy", i32 3}
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"tapir.loop.lowering.enabled"}

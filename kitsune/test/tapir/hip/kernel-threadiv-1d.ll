; Check that the induction variable for a 1D launch is computed correctly and
; bypasses the body of the loop if out of bounds
;
; RUN: opt --tapir=hip --tapir-hip-arch=gfx906 \
; RUN:     --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:     -passes='loop-spawning' -S %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: define {{.+}}(
; CHECK-SAME: i64 {{[^%]*}}%[[LB:[^,]+]],
; CHECK-SAME: i64 {{[^%]*}}%[[UB:[^,]+]],
; CHECK-SAME: i64 {{[^)]+}})
; CHECK-SAME: #[[ATTRS:[0-9]+]]
; CHECK: %[[WITEM:.+]] = {{.*}}call i32 @llvm.amdgcn.workitem.id.x()
; CHECK: %[[TID:.+]] = zext i32 %[[WITEM]] to i64
; CHECK: %[[BDIM:.+]] = {{.*}}call i64 @__ockl_get_local_size(i32 0)
; CHECK: %[[WGRP:.+]] = {{.*}}call i32 @llvm.amdgcn.workgroup.id.x()
; CHECK: %[[BIDX:.+]] = zext i32 %[[WGRP]] to i64
; CHECK: %[[BOFF:.+]] = mul i64 %[[BIDX]], %[[BDIM]]
; CHECK: %[[TIV:.+]] = add i64 %[[TID]], %[[BOFF]]
; CHECK: %[[COND:.+]] = icmp uge i64 %[[TIV]], %[[UB]]
; CHECK-NEXT: br i1 %[[COND]], label %[[BBEXIT:[^,]+]],
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

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.spawn.strategy", i32 3}
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"tapir.loop.lowering.enabled"}

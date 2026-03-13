; Check that the kernel generated immediately after lowering of a tapir loop
; nest of depth 1 is as expected. In this case, a single loop will be present
; in the kernel body. The start index of the loop will be an index computed
; from the hip functions that return threadIdx.x and blockIdx.x. The end will
; depend on the grainsize, but we test for the expected grainsize used in a
; different test.
;
; RUN: opt --tapir=hip -passes='loop-spawning' %s \
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
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %a.i = getelementptr i64, ptr %a, i64 %i
  store i64 %i, ptr %a.i
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
!1 = !{!"tapir.loop.target", i32 4}
!2 = !{!"tapir.loop.spawn.strategy", i32 3}
!3 = !{!"tapir.loop.lowering.enabled"}
!4 = !{!"tapir.loop.perfect.depth", i32 1}
!5 = !{!"tapir.loop.perfect.level", i32 1}

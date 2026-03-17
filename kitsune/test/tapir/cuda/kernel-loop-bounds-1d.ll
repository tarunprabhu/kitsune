; Check that the loop bounds of the loops are replaced correctly in the
; kernel function generated from a tapir loop nest of depth 1.
;
; NOTE: The upper bound is determined by the grainsize. We deliberately do not
; check for the actual grainsize here. That will be tested elsewhere.
;
; RUN: opt --tapir=cuda -passes='loop-spawning' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: define
; CHECK-NEXT: [[PH_X:.+]]:
; CHECK: %[[IVBEG_X:.+]] = zext i32 %{{.+}} to i64
; CHECK: %[[IVEND_X:.+]] = add i64 %[[IVBEG_X]]
;
; CHECK: [[HEADER_X:.+]]:
; CHECK: %[[IV_X:.+]] = phi i64
; CHECK-SAME: [ %[[IVBEG_X]], %[[PH_X]] ]
; CHECK-SAME: [ %[[IVNEXT_X:.+]], %[[LATCH_X:.+]] ]
;
; CHECK: [[LATCH_X]]:
; CHECK: %[[IVNEXT_X:.+]] = add {{.*}}i64 %[[IV_X]]
; CHECK: %[[IVCOND_X:.+]] = icmp eq i64 %[[IVNEXT_X]], %[[IVEND_X]]
; CHECK: br i1 %[[IVCOND_X]], label %[[EXIT:.+]], label %[[HEADER_X]]
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
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"tapir.loop.lowering.enabled"}
!4 = !{!"tapir.loop.perfect.depth", i32 1}
!5 = !{!"tapir.loop.perfect.level", i32 1}

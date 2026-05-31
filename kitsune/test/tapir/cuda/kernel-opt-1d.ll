; Check that when the standard sequence of optimization passes are run on the
; device module generated from a tapir loop nest of depth 1, the results are
; as expected.
;
; ------------------------------------------------------------------------------
;
; We have to set the optimization level of tapir lowering to non-zero for it to
; work. However, we can override the optimization level used on the embedded
; bitcode module. The lowering only adjusts the bounds of the original tapir
; loop. Setting the optimization level to O0 retains this loop.
;
; RUN: opt --tapir=cuda -passes='loop-spawning,emb-optimize' %s \
; RUN:     -emb-O0 \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s --check-prefix=O0
;
; O0: define {{.+}} @__kitcuda_{{.+}}(i64
; O0: = phi i64
;
; ------------------------------------------------------------------------------
;
; If compiling with optimizations, the loop will be removed since the trip count
; is determined to be 1. This is because the 'cuda' tapir target hard-codes this
; value when generating the code. If the grain size is changed to be something
; other than 1, this will certainly change.
;
; RUN: opt --tapir=cuda -passes='loop-spawning,emb-optimize' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s --check-prefix=O2
;
; O2-NOT: = phi i64
; O2: define {{.+}} @__kitcuda_{{[^(]+}}(
; O2-SAME: i64 {{[^%]*}}%[[LB_X:[^,]+]],
; O2-SAME: i64 {{[^%]*}}%[[TC_X:[^,]+]],
; O2-SAME: ptr {{[^%]*}}%[[BUF:[^,]+]],
; O2-SAME: i64 {{[^%]*}}%[[N:[^)]+]])
; O2-SAME: {{.*}}#[[ATTRS:[0-9]+]]
; O2-NEXT: [[PH_X:.+]]:
; O2: %[[IVBEG_X:.+]] = zext i32 %{{.+}} to i64
; O2-NEXT: %[[IVCOND_X:.+]] = icmp ugt i64 %[[TC_X]], %[[IVBEG_X]]
; O2-NEXT: br i1 %[[IVCOND_X]], label %[[BODY:[^,]+]], label %[[EXIT:.+]]
; O2: [[BODY]]:
; O2-NEXT: %[[ARRIDX:.+]] = getelementptr {{.+}}, ptr %[[BUF]], i64 %[[IVBEG_X]]
; O2-NEXT: store i64 %[[N]], ptr %[[ARRIDX]]
; O2-NEXT: br label %[[EXIT]]
; O2: [[EXIT]]:
; O2-NEXT: ret void
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
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"tapir.loop.lowering.enabled"}
!4 = !{!"tapir.loop.perfect.depth", i32 1}
!5 = !{!"tapir.loop.perfect.level", i32 1}

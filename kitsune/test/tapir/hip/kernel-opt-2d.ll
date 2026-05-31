; Check that when the standard sequence of optimization passes are run on the
; device module generated from a tapir loop nest of depth 2, the results are
; as expected.
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
; O0: = phi i64
;
; ------------------------------------------------------------------------------
;
; If compiling with optimizations, the loop will be removed since the trip count
; is determined to be 1. This is because the 'hip' tapir target hard-codes this
; value when generating the code. If the grain size is changed to be something
; other than 1, this will certainly change.
;
; RUN: opt --tapir=hip -passes='loop-spawning,emb-optimize' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s --check-prefix=O2
;
; O2-NOT: = phi i64
; O2: define {{.+}} @__kithip_{{[^(]+}}(
; O2-SAME: i64 {{[^%]*}}%[[LB_Y:[^,]+]],
; O2-SAME: i64 {{[^%]*}}%[[TC_Y:[^,]+]],
; O2-SAME: i64 {{[^%]*}}%[[LB_X:[^,]+]],
; O2-SAME: i64 {{[^%]*}}%[[TC_X:[^,]+]],
; O2-SAME: ptr {{[^%]*}}%[[BUF:[^,]+]],
; O2-SAME: i64 {{[^%]*}}%[[N:[^)]+]])
; O2-SAME: #{{[0-9]+}}
;
; O2-NEXT: [[PH_Y:.+]]:
; O2: %[[IVBEG_Y:.+]] = zext i32 %{{.+}} to i64
; O2: %[[IVBEG_X:.+]] = zext i32 %{{.+}} to i64
; O2-NEXT: %[[IVCOND_Y:.+]] = icmp ugt i64 %[[TC_Y]], %[[IVBEG_Y]]
; O2-NEXT: %[[IVCOND_X:.+]] = icmp ugt i64 %[[TC_X]], %[[IVBEG_X]]
; O2-NEXT: %[[IVCOND:.+]] = and i1 %[[IVCOND_Y]], %[[IVCOND_X]]
; O2-NEXT: br i1 %[[IVCOND]], label %[[BODY:[^,]+]], label %[[EXIT:.+]]
;
; O2: [[BODY]]:
; O2: call void @ext2(ptr %[[BUF]], i64 %[[IVBEG_Y]], i64 %[[IVBEG_X]])
; O2-NEXT: br label %[[EXIT]]
;
; O2: [[EXIT]]:
; O2-NEXT: ret void
;
; ------------------------------------------------------------------------------

define void @pp(i64 %m, i64 %n, ptr %c) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.j = tail call token @llvm.syncregion.start()
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  call void @ext2(ptr %c, i64 %i, i64 %j)
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !1

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

declare void @ext2(ptr, i64, i64)

!0 = distinct !{!0, !2, !3, !4, !5, !6}
!1 = distinct !{!1, !2, !3, !7}
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"tapir.loop.spawn.strategy", i32 3}
!4 = !{!"tapir.loop.lowering.enabled"}
!5 = !{!"tapir.loop.perfect.depth", i32 2}
!6 = !{!"tapir.loop.perfect.level", i32 1}
!7 = !{!"tapir.loop.perfect.level", i32 2}

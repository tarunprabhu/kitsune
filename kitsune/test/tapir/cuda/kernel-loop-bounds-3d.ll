; Check that the loop bounds of the loops are replaced correctly in the
; kernel function generated from a tapir loop nest of depth 3.
;
; NOTE: The upper bound is determined by the grainsize. We deliberately do not
; check for the actual grainsize here. That will be tested elsewhere.
;
; RUN: opt --tapir=cuda -passes='loop-spawning' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | %kit-sort \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: define {{.*}}ptx_kernel
;
; CHECK: %[[IVBEG_Z:.+]] = zext i32 %{{.+}} to i64
; CHECK: %[[IVEND_Z:.+]] = add i64 %[[IVBEG_Z]]
; CHECK: %[[IVBEG_Y:.+]] = zext i32 %{{.+}} to i64
; CHECK: %[[IVEND_Y:.+]] = add i64 %[[IVBEG_Y]]
; CHECK: %[[IVBEG_X:.+]] = zext i32 %{{.+}} to i64
; CHECK: %[[IVEND_X:.+]] = add i64 %[[IVBEG_X]]
;
; CHECK: %[[IV_Z:.+]] = phi i64
; CHECK-SAME: [ %[[IVBEG_Z]], %{{[^]]+}} ]
; CHECK-SAME: [ %[[IVNEXT_Z:.+]], %[[LATCH_Z:.+]] ]
;
; CHECK: %[[IV_Y:.+]] = phi i64
; CHECK-SAME: [ %[[IVBEG_Y]], %{{[^]]+}} ]
; CHECK-SAME: [ %[[IVNEXT_Y:.+]], %[[LATCH_Y:.+]] ]
;
; CHECK: %[[IV_X:.+]] = phi i64
; CHECK-SAME: [ %[[IVBEG_X]], %{{[^]]+}} ]
; CHECK-SAME: [ %[[IVNEXT_X:.+]], %[[LATCH_X:.+]] ]
;
; CHECK: [[LATCH_X]]:
; CHECK: %[[IVNEXT_X:.+]] = add {{.*}}i64 %[[IV_X]]
; CHECK: %[[IVCOND_X:.+]] = icmp eq i64 %[[IVNEXT_X]], %[[IVEND_X]]
; CHECK: br i1 %[[IVCOND_X]], label %{{.+}}, label %{{.+}}
;
; CHECK: [[LATCH_Y]]:
; CHECK: %[[IVNEXT_Y:.+]] = add {{.*}}i64 %[[IV_Y]]
; CHECK: %[[IVCOND_Y:.+]] = icmp eq i64 %[[IVNEXT_Y]], %[[IVEND_Y]]
; CHECK: br i1 %[[IVCOND_Y]], label %{{.+}}, label %{{.+}}
;
; CHECK: [[LATCH_Z]]:
; CHECK: %[[IVNEXT_Z:.+]] = add {{.*}}i64 %[[IV_Z]]
; CHECK: %[[IVCOND_Z:.+]] = icmp eq i64 %[[IVNEXT_Z]], %[[IVEND_Z]]
; CHECK: br i1 %[[IVCOND_Z]], label %[[EXIT:.+]], label %{{.+}}
;
; CHECK: [[EXIT]]:
; CHECK-NEXT: ret void

define void @ppp(i64 %m, i64 %n, i64 %p, ptr %c) {
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
  %syncreg.k = tail call token @llvm.syncregion.start()
  br label %for.k.header

for.k.header:
  %k = phi i64 [ 0, %for.j.body ], [ %inc.k, %for.k.latch ]
  detach within %syncreg.k, label %for.k.body, label %for.k.latch

for.k.body:
  call void @ext3(ptr %c, i64 %i, i64 %j, i64 %k)
  reattach within %syncreg.k, label %for.k.latch

for.k.latch:
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !2

for.k.exit:
  sync within %syncreg.k, label %for.k.end

for.k.end:
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

declare void @ext3(ptr, i64, i64, i64)

!0 = distinct !{!0, !3, !4, !5, !6, !7}
!1 = distinct !{!1, !3, !4, !8}
!2 = distinct !{!2, !3, !4, !9}
!3 = !{!"tapir.loop.target", i32 2}
!4 = !{!"tapir.loop.spawn.strategy", i32 3}
!5 = !{!"tapir.loop.lowering.enabled"}
!6 = !{!"tapir.loop.perfect.depth", i32 3}
!7 = !{!"tapir.loop.perfect.level", i32 1}
!8 = !{!"tapir.loop.perfect.level", i32 2}
!9 = !{!"tapir.loop.perfect.level", i32 3}

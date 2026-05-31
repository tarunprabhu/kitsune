; Check that the induction variable in a kernel generated from a loop nest of
; depth 3 is computed correctly. The loop nest should be bypassed if the
; computed value is out of bounds.
;
; RUN: opt --tapir=cuda -passes='loop-spawning' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: define
; CHECK-SAME: i64 {{[^%]*}}%[[LB_Z:[^,]+]],
; CHECK-SAME: i64 {{[^%]*}}%[[TC_Z:[^,]+]],
; CHECK-SAME: i64 {{[^%]*}}%[[LB_Y:[^,]+]],
; CHECK-SAME: i64 {{[^%]*}}%[[TC_Y:[^,]+]],
; CHECK-SAME: i64 {{[^%]*}}%[[LB_X:[^,]+]],
; CHECK-SAME: i64 {{[^%]*}}%[[TC_X:[^,]+]],
; CHECK-SAME: #{{[0-9]+}}
;
; CHECK: %[[TID_Z:.+]] = {{.*}}call i32 @llvm.kit.gpu.thread.id.z(i32 2)
; CHECK: %[[BID_Z:.+]] = {{.*}}call i32 @llvm.kit.gpu.block.id.z(i32 2)
; CHECK: %[[BSZ_Z:.+]] = {{.*}}call i32 @llvm.kit.gpu.block.size.z(i32 2)
; CHECK: %[[OFF_Z:.+]] = mul i32 %[[BSZ_Z]], %[[BID_Z]]
; CHECK: %[[IVBEG32_Z:.+]] = add i32 %[[OFF_Z]], %[[TID_Z]]
; CHECK: %[[IVBEG_Z:.+]] = zext i32 %[[IVBEG32_Z]] to i64
;
; CHECK: %[[TID_Y:.+]] = {{.*}}call i32 @llvm.kit.gpu.thread.id.y(i32 2)
; CHECK: %[[BID_Y:.+]] = {{.*}}call i32 @llvm.kit.gpu.block.id.y(i32 2)
; CHECK: %[[BSZ_Y:.+]] = {{.*}}call i32 @llvm.kit.gpu.block.size.y(i32 2)
; CHECK: %[[OFF_Y:.+]] = mul i32 %[[BSZ_Y]], %[[BID_Y]]
; CHECK: %[[IVBEG32_Y:.+]] = add i32 %[[OFF_Y]], %[[TID_Y]]
; CHECK: %[[IVBEG_Y:.+]] = zext i32 %[[IVBEG32_Y]] to i64
;
; CHECK: %[[TID_X:.+]] = {{.*}}call i32 @llvm.kit.gpu.thread.id.x(i32 2)
; CHECK: %[[BID_X:.+]] = {{.*}}call i32 @llvm.kit.gpu.block.id.x(i32 2)
; CHECK: %[[BSZ_X:.+]] = {{.*}}call i32 @llvm.kit.gpu.block.size.x(i32 2)
; CHECK: %[[OFF_X:.+]] = mul i32 %[[BSZ_X]], %[[BID_X]]
; CHECK: %[[IVBEG32_X:.+]] = add i32 %[[OFF_X]], %[[TID_X]]
; CHECK: %[[IVBEG_X:.+]] = zext i32 %[[IVBEG32_X]] to i64
;
; CHECK: %[[IVCOND_Z:.+]] = icmp ult i64 %[[IVBEG_Z]], %[[TC_Z]]
; CHECK: %[[IVCOND_Y:.+]] = icmp ult i64 %[[IVBEG_Y]], %[[TC_Y]]
; CHECK: %[[IVCOND_X:.+]] = icmp ult i64 %[[IVBEG_X]], %[[TC_X]]
; CHECK: %[[IVCOND_TMP:.+]] = and i1 %[[IVCOND_Z]], %[[IVCOND_Y]]
; CHECK: %[[IVCOND:.+]] = and i1 %[[IVCOND_TMP]], %[[IVCOND_X]]
;
; CHECK-NEXT: br i1 %[[IVCOND]], label %[[HEADER:[^,]+]], label %[[EXIT:.+]]
; CHECK: [[HEADER]]:
; CHECK-NEXT: phi i64
; CHECK: [[EXIT]]:
; CHECK-NEXT: ret void

define void @pp(i64 %m, i64 %n, i64 %p, ptr %c) {
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
  %arrayidx = getelementptr i64, ptr %c, i64 %j
  %product = mul i64 %m, %n
  store i64 %product, ptr %arrayidx
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

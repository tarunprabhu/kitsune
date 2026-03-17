; Check that the induction variable in a kernel generated from a loop nest of
; depth 2 is computed correctly. The loop nest should be bypassed if the
; computed value is out of bounds.
;
; RUN: opt --tapir=cuda -passes='loop-spawning' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: define
; CHECK-SAME: i64 {{[^%]*}}%[[LB_Y:[^,]+]],
; CHECK-SAME: i64 {{[^%]*}}%[[TC_Y:[^,]+]],
; CHECK-SAME: i64 {{[^%]*}}%[[LB_X:[^,]+]],
; CHECK-SAME: i64 {{[^%]*}}%[[TC_X:[^,]+]],
; CHECK-SAME: #{{[0-9]+}}
;
; CHECK: %[[TID_Y:.+]] = {{.*}}call i32 @llvm.kit.gpu.thread.id.y()
; CHECK: %[[BID_Y:.+]] = {{.*}}call i32 @llvm.kit.gpu.block.id.y()
; CHECK: %[[BSZ_Y:.+]] = {{.*}}call i32 @llvm.kit.gpu.block.size.y()
; CHECK: %[[OFF_Y:.+]] = mul i32 %[[BSZ_Y]], %[[BID_Y]]
; CHECK: %[[IVBEG32_Y:.+]] = add i32 %[[OFF_Y]], %[[TID_Y]]
; CHECK: %[[IVBEG_Y:.+]] = zext i32 %[[IVBEG32_Y]] to i64
;
; CHECK: %[[TID_X:.+]] = {{.*}}call i32 @llvm.kit.gpu.thread.id.x()
; CHECK: %[[BID_X:.+]] = {{.*}}call i32 @llvm.kit.gpu.block.id.x()
; CHECK: %[[BSZ_X:.+]] = {{.*}}call i32 @llvm.kit.gpu.block.size.x()
; CHECK: %[[OFF_X:.+]] = mul i32 %[[BSZ_X]], %[[BID_X]]
; CHECK: %[[IVBEG32_X:.+]] = add i32 %[[OFF_X]], %[[TID_X]]
; CHECK: %[[IVBEG_X:.+]] = zext i32 %[[IVBEG32_X]] to i64
;
; CHECK: %[[IVCOND_Y:.+]] = icmp ult i64 %[[IVBEG_Y]], %[[TC_Y]]
; CHECK: %[[IVCOND_X:.+]] = icmp ult i64 %[[IVBEG_X]], %[[TC_X]]
; CHECK: %[[IVCOND:.+]] = and i1 %[[IVCOND_Y]], %[[IVCOND_X]]
;
; CHECK-NEXT: br i1 %[[IVCOND]], label %[[HEADER:[^,]+]], label %[[EXIT:.+]]
; CHECK: [[HEADER]]:
; CHECK-NEXT: phi i64
; CHECK: [[EXIT]]:
; CHECK-NEXT: ret void

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
  %arrayidx = getelementptr i64, ptr %c, i64 %j
  %product = mul i64 %m, %n
  store i64 %product, ptr %arrayidx
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

!0 = distinct !{!0, !2, !3, !4, !5, !6}
!1 = distinct !{!1, !2, !3, !7}
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"tapir.loop.spawn.strategy", i32 3}
!4 = !{!"tapir.loop.lowering.enabled"}
!5 = !{!"tapir.loop.perfect.depth", i32 2}
!6 = !{!"tapir.loop.perfect.level", i32 1}
!7 = !{!"tapir.loop.perfect.level", i32 2}

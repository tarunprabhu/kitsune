; Check that the general structure of a kernel generated immediately after
; lowering of a tapir loop nest of depth 3 is as expected. In this case, three
; loops will be present in the kernel body.
;
; RUN: opt --tapir=cuda -passes='loop-spawning' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | %kit-sort \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: define {{.*}}ptx_kernel
;
; CHECK: [[PH_Z:.+]]:
; CHECK: [[HEADER_Z:.+]]:
; CHECK-NEXT: phi i64
; CHECK-SAME: [ {{[^,]+}}, %[[PH_Z]] ]
; CHECK-SAME: [ {{[^,]+}}, %[[LATCH_Z:.+]] ]
; CHECK-NEXT: br label %[[BODY_Z:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY_Z]]:
; CHECK-NEXT: br label %[[HEADER_Y:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER_Y:.+]]:
; CHECK-NEXT: phi i64
; CHECK-SAME: [ {{[^,]+}}, %[[BODY_Z]] ]
; CHECK-SAME: [ {{[^,]+}}, %[[LATCH_Y:.+]] ]
; CHECK-NEXT: br label %[[BODY_Y:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY_Y]]:
; CHECK-NEXT: br label %[[HEADER_X:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER_X]]:
; CHECK-NEXT: phi i64
; CHECK-SAME: [ {{[^,]+}}, %[[BODY_Y]] ]
; CHECK-SAME: [ {{[^,]+}}, %[[LATCH_X:.+]] ]
; CHECK-NEXT: br label %[[BODY_X:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY_X]]:
; CHECK-NEXT: call void @ext3
; CHECK-NEXT: br label %[[LATCH_X]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH_X]]:
; CHECK: br i1 %{{.+}}, label %[[EXIT_X:.+]], label %[[HEADER_X]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT_X]]:
; CHECK-NEXT: br label %[[END_X:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[END_X:.+]]:
; CHECK-NEXT: br label %[[LATCH_Y]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH_Y]]:
; CHECK: br i1 %{{.+}}, label %[[EXIT_Y:.+]], label %[[HEADER_Y]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT_Y]]:
; CHECK-NEXT: br label %[[END_Y:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[END_Y:.+]]:
; CHECK-NEXT: br label %[[LATCH_Z]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH_Z]]:
; CHECK: br i1 %{{.+}}, label %[[EXIT_Z:.+]], label %[[HEADER_Z]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT_Z]]:
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

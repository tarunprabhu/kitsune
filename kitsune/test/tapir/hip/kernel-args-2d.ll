; Check that the number and type of arguments in the generated kernel function
; are as expected when the source tapir loop nest has depth 2.
;
; RUN: opt --tapir=hip -passes='loop-spawning' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: define {{.*}}amdgpu_kernel void
; CHECK-SAME: (i64 %zero.y{{[^,]*}},
; CHECK-SAME: i64 %tc.y{{[^,]*}},
; CHECK-SAME: i64 %zero.x{{[^,]*}},
; CHECK-SAME: i64 %tc.x{{[^,]*}},
; CHECK-SAME: ptr{{[^%]*}} %c{{[^,]*}},
; CHECK-SAME: i64 %m{{[^,]*}},
; CHECK-SAME: i64 %n{{[^,]*}})
; CHECK-SAME: #{{[0-9]+}}

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
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"tapir.loop.spawn.strategy", i32 3}
!4 = !{!"tapir.loop.lowering.enabled"}
!5 = !{!"tapir.loop.perfect.depth", i32 2}
!6 = !{!"tapir.loop.perfect.level", i32 1}
!7 = !{!"tapir.loop.perfect.level", i32 2}

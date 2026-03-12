; All perfectly nested tapir loops intended to be lowered with the 'hip' tapir
; target must be canonical.
;
; RUN: not opt --tapir=nolo --passes=kit-verify-prelower %s 2>&1 \
; RUN:     -disable-output \
; RUN:     | FileCheck %s
;

; CHECK: tapir loop for GPU is not canonical
; CHECK-NEXT: from loop 'p.loop.i'
; CHECK-NEXT: from function 'p'
;
; forall (i ...)
define void @p(ptr %a, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 10, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
  %a.i = getelementptr i64, ptr %a, i64 %i
  store i64 %i, ptr %a.i
  reattach within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.sync, label %for.i.header, !llvm.loop !0

for.i.sync:
  sync within %syncreg, label %for.i.exit

for.i.exit:
  ret void
}

; CHECK: tapir loop for GPU is not canonical
; CHECK-NEXT: from loop 'pp.loop.j'
; CHECK-NEXT: from function 'pp'
;
; forall (i ...)
;   forall (j ...)
define void @pp(i64 %m, i64 %n) {
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
  %j = phi i64 [ 10, %for.i.body ], [ %inc.j, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !4

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !3

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}


!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.target", i32 4}
!2 = !{!"loop.name", !"p.loop.i"}
!3 = distinct !{!3, !1, !5}
!4 = distinct !{!4, !1, !6}
!5 = !{!"loop.name", !"pp.loop.i"}
!6 = !{!"loop.name", !"pp.loop.j"}
!7 = distinct !{!7, !1, !9}
!8 = distinct !{!8}
!9 = !{!"loop.name", !"ps.loop.i"}

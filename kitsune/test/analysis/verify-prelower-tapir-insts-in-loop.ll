; If a loop is not determined to be a tapir loop, warn if detach/reattach
; instructions are found within it.
;
; RUN: not opt --tapir=nolo -passes='kit-verify-prelower' %s 2>&1 \
; RUN:     -disable-output \
; RUN:     | FileCheck %s

; CHECK: tapir instructions outside tapir loops are not yet supported
; CHECK: tapir instructions outside tapir loops are not yet supported
;
; for (i ... m)
define void @f1(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
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

; CHECK: tapir instructions outside tapir loops are not yet supported
; CHECK: tapir instructions outside tapir loops are not yet supported
;
; forall (i ... m)
;   for (j ... n)
;     for (k ... p)
define void @f3(i64 %m, i64 %n, i64 %p) {
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
  br label %for.k.body

for.k.body:
  %k = phi i64 [ 0, %for.j.body ], [ %inc.k, %for.k.body ]
  %inc.k = add nuw nsw i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k.body, !llvm.loop !3

for.k.exit:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add nuw nsw i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !2

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add nuw nsw i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !1

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

!0 = distinct !{!0, !4}
!1 = distinct !{!1, !5, !7}
!2 = distinct !{!2, !6}
!3 = distinct !{!3}
!4 = !{!"loop.name", !"f1.loop.i"}
!5 = !{!"loop.name", !"f3.loop.i"}
!6 = !{!"loop.name", !"f3.loop.j"}
!7 = !{!"tapir.loop.target", i32 1024}

; Check that the serialize tapir loops pass correctly serializes tapir loops
; that are imperfectly nested in a tapir loop nest. In the current
; implementation, this results in the innermost loop nests being serialized.
;
; RUN: opt --tapir=cuda -passes="kit-serialize" -S %s \
; RUN:   | FileCheck %s
;
; CHECK: %syncreg.i = tail call token @llvm.syncregion.start()
; CHECK: %i = phi i64
; CHECK: detach within %syncreg.i
; CHECK-NOT: tail call token @llvm.syncregion.start()
; CHECK: %j = phi i64
; CHECK: %k = phi i64
; CHECK-NOT: detach within %syncreg.j
; CHECK-NOT: reattach within %syncreg.j
; CHECK-NOT: sync within %syncreg.j
; CHECK: reattach within %syncreg.i
; CHECK: sync within %syncreg.i
;
; CHECK: %syncreg2.i = tail call token @llvm.syncregion.start()
; CHECK: %i2 = phi i64
; CHECK: detach within %syncreg2.i
; CHECK: %j2 = phi i64
; CHECK-NOT: tail call token @llvm.syncregion.start()
; CHECK: %k2 = phi i64
; CHECK-NOT: detach within %syncreg.k
; CHECK-NOT: tail call token @llvm.syncregion.start()
; CHECK: %l = phi i64
; CHECK-NOT: detach within %syncreg.l
; CHECK-NOT: reattach within %syncreg.l
; CHECK-NOT: sync within %syncreg.l
; CHECK-NOT: reattach within %syncreg2.k
; CHECK-NOT: sync within %syncreg2.k
; CHECK: reattach within %syncreg2.i
; CHECK: sync within %syncreg2.i
;
; This contains two loop nests
;
; forall (i ...)
;   for (j ...)
;     forall (k ...)
;
; forall (i ...)
;   for (j ...)
;     forall (k ...)
;       forall (l ...)
;
define dso_local void @f(i64 %m, i64 %n, i64 %p, i64 %q) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  %cmp.m.not = icmp eq i64 %m, 0
  %cmp.n.not = icmp eq i64 %n, 0
  %cmp.p.not = icmp eq i64 %p, 0
  %cmp.q.not = icmp eq i64 %q, 0
  br i1 %cmp.m.not, label %for.i.exit, label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.k = tail call token @llvm.syncregion.start()
  br i1 %cmp.n.not, label %for.j.exit, label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j.latch ]
  br i1 %cmp.p.not, label %for.k.exit, label %for.k.header

for.k.header:
  %k = phi i64 [ 0, %for.j.header ], [ %inc.k, %for.k.latch ]
  detach within %syncreg.k, label %for.k.body, label %for.k.latch

for.k.body:
  reattach within %syncreg.k, label %for.k.latch

for.k.latch:
  %inc.k = add i64 %k, 1
  %exitcond.k.not = icmp eq i64 %inc.k, %p
  br i1 %exitcond.k.not, label %for.k.exit, label %for.k.header, !llvm.loop !0

for.k.exit:
  sync within %syncreg.k, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %exitcond.j.not = icmp eq i64 %inc.j, %n
  br i1 %exitcond.j.not, label %for.j.exit, label %for.j.header, !llvm.loop !2

for.j.exit:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %exitcond.i.not = icmp eq i64 %inc.i, %m
  br i1 %exitcond.i.not, label %for.i.exit, label %for.i.header, !llvm.loop !3

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  br label %entry2

entry2:
  %syncreg2.i = tail call token @llvm.syncregion.start()
  br i1 %cmp.m.not, label %for2.i.exit, label %for2.i.header

for2.i.header:
  %i2 = phi i64 [ 0, %entry2 ], [ %inc.i2, %for2.i.latch ]
  detach within %syncreg2.i, label %for2.i.body, label %for2.i.latch

for2.i.body:
  br i1 %cmp.n.not, label %for2.j.exit, label %for2.j.header

for2.j.header:
  %j2 = phi i64 [ 0, %for2.i.body ], [ %inc.j2, %for2.j.latch ]
  br label %for2.j.body

for2.j.body:
  %syncreg2.k = tail call token @llvm.syncregion.start()
  br i1 %cmp.p.not, label %for2.k.exit, label %for2.k.header

for2.k.header:
  %k2 = phi i64 [ 0, %for2.j.body ], [ %inc.k2, %for2.k.latch ]
  detach within %syncreg2.k, label %for2.k.body, label %for2.k.latch

for2.k.body:
  %syncreg2.l = tail call token @llvm.syncregion.start()
  br i1 %cmp.q.not, label %for2.l.exit, label %for2.l.header

for2.l.header:
  %l = phi i64 [ 0, %for2.k.body ], [ %inc.l, %for2.l.latch ]
  detach within %syncreg2.l, label %for2.l.body, label %for2.l.latch

for2.l.body:
  reattach within %syncreg2.l, label %for2.l.latch

for2.l.latch:
  %inc.l = add i64 %l, 1
  %exitcond.l.not = icmp eq i64 %inc.l, %q
  br i1 %exitcond.l.not, label %for2.l.exit, label %for2.l.header, !llvm.loop !4

for2.l.exit:
  sync within %syncreg2.l, label %for2.l.end

for2.l.end:
  reattach within %syncreg2.k, label %for2.k.latch

for2.k.latch:
  %inc.k2 = add i64 %k2, 1
  %exitcond.k2.not = icmp eq i64 %inc.k2, %p
  br i1 %exitcond.k2.not, label %for2.k.exit, label %for2.k.header, !llvm.loop !5

for2.k.exit:
  sync within %syncreg2.k, label %for2.j.latch

for2.j.latch:
  %inc.j2 = add i64 %j2, 1
  %exitcond.j2.not = icmp eq i64 %inc.j2, %n
  br i1 %exitcond.j2.not, label %for2.j.exit, label %for2.j.header, !llvm.loop !6

for2.j.exit:
  reattach within %syncreg2.i, label %for2.i.latch

for2.i.latch:
  %inc.i2 = add nuw i64 %i2, 1
  %exitcond.i2.not = icmp eq i64 %inc.i2, %m
  br i1 %exitcond.i2.not, label %for2.i.exit, label %for2.i.header, !llvm.loop !7

for2.i.exit:
  sync within %syncreg2.i, label %for2.i.end

for2.i.end:
  ret void

}

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 2}
!2 = distinct !{!2}
!3 = distinct !{!3, !1, !8, !9}
!4 = distinct !{!4, !1}
!5 = distinct !{!5, !1}
!6 = distinct !{!6}
!7 = distinct !{!7, !1, !8, !9}
!8 = !{!"tapir.loop.perfect.depth", i32 1}
!9 = !{!"tapir.loop.perfect.level", i32 1}

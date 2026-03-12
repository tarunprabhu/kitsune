; Check that tapir loops at a perfect nesting level greater than 3 are
; serialized by the kit-serialize pass.
;
; RUN: opt -passes="kit-serialize" -S %s \
; RUN:   | FileCheck %s
;
; CHECK: %syncreg.i = tail call token @llvm.syncregion.start()
; CHECK: %i = phi i64
; CHECK: detach within %syncreg.i
; CHECK: %syncreg.j = tail call token @llvm.syncregion.start()
; CHECK: %j = phi i64
; CHECK: detach within %syncreg.j
; CHECK: %syncreg.k = tail call token @llvm.syncregion.start()
; CHECK: %k = phi i64
; CHECK: detach within %syncreg.k
; CHECK-NOT: %syncreg.l = tail call token @llvm.syncregion.start()
; CHECK: %l = phi i64
; CHECK-NOT: detach within %syncreg.l
; CHECK-NOT: reattach within %syncreg.l
; CHECK-NOT: sync within %syncreg.l
; CHECK: !llvm.loop ![[LOOP_L:[0-9]+]]
; CHECK: reattach within %syncreg.k
; CHECK: !llvm.loop ![[LOOP_K:[0-9]+]]
; CHECK: sync within %syncreg.k
; CHECK: reattach within %syncreg.j
; CHECK: !llvm.loop ![[LOOP_J:[0-9]+]]
; CHECK: sync within %syncreg.j
; CHECK: reattach within %syncreg.i
; CHECK: !llvm.loop ![[LOOP_I:[0-9]+]]
; CHECK: sync within %syncreg.i
; CHECK: ret void
;
; CHECK-DAG: ![[TARGET:[0-9]+]] = !{!"tapir.loop.target", i32 2}
; CHECK-DAG: ![[L3:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 3}
; CHECK-DAG: ![[L2:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 2}
; CHECK-DAG: ![[L1:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 1}
; CHECK-DAG: ![[D3:[0-9]+]] = !{!"tapir.loop.perfect.depth", i32 3}
; CHECK-DAG: ![[LOOP_L]] = distinct !{![[LOOP_L]]}
; CHECK-DAG: ![[LOOP_K]] = distinct !{![[LOOP_K]], ![[TARGET]], ![[L3]]}
; CHECK-DAG: ![[LOOP_J]] = distinct !{![[LOOP_J]], ![[TARGET]], ![[L2]]}
; CHECK-DAG: ![[LOOP_I]] = distinct !{![[LOOP_I]], ![[TARGET]], ![[L1]], ![[D3]]}

; forall (i ...)
;   forall (j ...)
;     forall (k ...)
;       forall (l ...)
define void @pppp(i64 %m, i64 %n, i64 %p, i64 %q) {
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
  %syncreg.l = tail call token @llvm.syncregion.start()
  br label %for.l.header

for.l.header:
  %l = phi i64 [ 0, %for.k.body ], [ %inc.l, %for.l.latch ]
  detach within %syncreg.l, label %for.l.body, label %for.l.latch

for.l.body:
  reattach within %syncreg.l, label %for.l.latch

for.l.latch:
  %inc.l = add i64 %l, 1
  %cmp.l = icmp eq i64 %inc.l, %q
  br i1 %cmp.l, label %for.l.exit, label %for.l.header, !llvm.loop !0

for.l.exit:
  sync within %syncreg.l, label %for.l.end

for.l.end:
  reattach within %syncreg.k, label %for.k.latch

for.k.latch:
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !1

for.k.exit:
  sync within %syncreg.k, label %for.k.end

for.k.end:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !2

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

!kit.module.loops.annotated = !{}

!0 = distinct !{!0, !4, !9}
!1 = distinct !{!1, !4, !8}
!2 = distinct !{!2, !4, !7}
!3 = distinct !{!3, !4, !6, !5}
!4 = !{!"tapir.loop.target", i32 2}
!5 = !{!"tapir.loop.perfect.depth", i32 4}
!6 = !{!"tapir.loop.perfect.level", i32 1}
!7 = !{!"tapir.loop.perfect.level", i32 2}
!8 = !{!"tapir.loop.perfect.level", i32 3}
!9 = !{!"tapir.loop.perfect.level", i32 4}

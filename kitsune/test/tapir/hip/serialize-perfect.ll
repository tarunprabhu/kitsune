; Check that the serialize tapir loops pass does not serialize any loops when
; all loop nests in the function are perfect.
;
; RUN: opt -passes="kit-serialize" -S %s \
; RUN:   | FileCheck %s

; CHECK-LABEL: @p
; CHECK: %syncreg.i = tail call token @llvm.syncregion.start()
; CHECK: %i = phi i64
; CHECK: detach within %syncreg.i
; CHECK: reattach within %syncreg.i
; CHECK: sync within %syncreg.i
; CHECK: ret void
;
; forall (i ...)
define void @p(i64 %n) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.sync, label %for.i.header, !llvm.loop !0

for.i.sync:
  sync within %syncreg.i, label %for.i.exit

for.i.exit:
  ret void
}

; CHECK-LABEL: @pp
; CHECK: %syncreg.i = tail call token @llvm.syncregion.start()
; CHECK: %i = phi i64
; CHECK: detach within %syncreg.i
; CHECK: %syncreg.j = tail call token @llvm.syncregion.start()
; CHECK: %j = phi i64
; CHECK: detach within %syncreg.j
; CHECK: reattach within %syncreg.j
; CHECK: sync within %syncreg.j
; CHECK: reattach within %syncreg.i
; CHECK: sync within %syncreg.i
; CHECK: ret void
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
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
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
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !2

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

; CHECK-LABEL: @pp
; CHECK: %syncreg.i = tail call token @llvm.syncregion.start()
; CHECK: %i = phi i64
; CHECK: detach within %syncreg.i
; CHECK: %syncreg.j = tail call token @llvm.syncregion.start()
; CHECK: %j = phi i64
; CHECK: detach within %syncreg.j
; CHECK: %syncreg.k = tail call token @llvm.syncregion.start()
; CHECK: detach within %syncreg.k
; CHECK: reattach within %syncreg.k
; CHECK: sync within %syncreg.k
; CHECK: reattach within %syncreg.j
; CHECK: sync within %syncreg.j
; CHECK: reattach within %syncreg.i
; CHECK: sync within %syncreg.i
; CHECK: ret void
;
; forall (i ...)
;   forall (j ...)
;     forall (k ...)
define void @ppp(i64 %m, i64 %n, i64 %p) {
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
  %k = phi i64 [0, %for.j.body ], [ %inc.k, %for.k.latch ]
  detach within %syncreg.k, label %for.k.body, label %for.k.latch

for.k.body:
  reattach within %syncreg.k, label %for.k.latch

for.k.latch:
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !3

for.k.exit:
  sync within %syncreg.k, label %for.k.end

for.k.end:
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
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !5

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

!kit.module.loops.annotated = !{}

!0 = distinct !{!0, !6, !7, !10}
!1 = distinct !{!1, !6, !11}
!2 = distinct !{!2, !6, !8, !10}
!3 = distinct !{!3, !6, !12}
!4 = distinct !{!4, !6, !11}
!5 = distinct !{!5, !6, !9, !10}
!6 = !{!"tapir.loop.target", i32 4}
!7 = !{!"tapir.loop.perfect.depth", i32 1}
!8 = !{!"tapir.loop.perfect.depth", i32 2}
!9 = !{!"tapir.loop.perfect.depth", i32 3}
!10 = !{!"tapir.loop.perfect.level", i32 1}
!11 = !{!"tapir.loop.perfect.level", i32 2}
!12 = !{!"tapir.loop.perfect.level", i32 3}

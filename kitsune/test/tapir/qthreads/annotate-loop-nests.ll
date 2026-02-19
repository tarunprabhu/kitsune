; Check that the tapir loop annotator pass annotates loops correctly. Every
; function here contains a single, imperfect loop nest.
;
; RUN: opt -passes="function(loop-simplify),kit-annotate-tapir-loops" \
; RUN:     --tapir=qthreads -S %s \
; RUN:     | FileCheck %s

; CHECK-LABEL: @pep
; CHECK: !llvm.loop ![[PEP_J:[0-9]+]]
; CHECK: !llvm.loop ![[PEP_I:[0-9]+]]
;
; forall (i ...) {
;   expr
;   forall (j ...)
;     ;
; }
define void @pep(i64 %m, i64 %n) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  %cmp.m.not = icmp eq i64 %m, 0
  %cmp.n.not = icmp eq i64 %n, 0
  br i1 %cmp.m.not, label %for.i.exit, label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.j = tail call token @llvm.syncregion.start()
  %0 = add i64 %m, %n
  br i1 %cmp.n.not, label %for.j.exit, label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %exitcond.j.not = icmp eq i64 %inc.j, %n
  br i1 %exitcond.j.not, label %for.j.exit, label %for.j.header, !llvm.loop !1

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %exitcond.i.not = icmp eq i64 %inc.i, %m
  br i1 %exitcond.i.not, label %for.i.exit, label %for.i.header, !llvm.loop !2

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

; CHECK-LABEL: @pepp
; CHECK: llvm.loop ![[PEPP_K:[0-9]+]]
; CHECK: llvm.loop ![[PEPP_J:[0-9]+]]
; CHECK: llvm.loop ![[PEPP_I:[0-9]+]]
;
; forall (i ...) {
;   expr;
;   forall (j ...) {
;     forall (k ...)
;       ...
;   }
; }
;
define void @pepp(i64 %m, i64 %n, i64 %p) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  %cmp.m.not = icmp eq i64 %m, 0
  %cmp.n.not = icmp eq i64 %n, 0
  %cmp.p.not = icmp eq i64 %p, 0
  br i1 %cmp.m.not, label %for.i.exit, label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.j = tail call token @llvm.syncregion.start()
  %0 = add i64 %m, %n
  br i1 %cmp.n.not, label %for.j.exit, label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  %syncreg.k = tail call token @llvm.syncregion.start()
  br i1 %cmp.p.not, label %for.k.exit, label %for.k.header

for.k.header:
  %k = phi i64 [0, %for.j.body ], [ %inc.k, %for.k.latch ]
  detach within %syncreg.k, label %for.k.body, label %for.k.latch

for.k.body:
  reattach within %syncreg.k, label %for.k.latch

for.k.latch:
  %inc.k = add i64 %k, 1
  %exitcond.k.not = icmp eq i64 %inc.k, %p
  br i1 %exitcond.k.not, label %for.k.exit, label %for.k.header, !llvm.loop !3

for.k.exit:
  sync within %syncreg.k, label %for.k.end

for.k.end:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %exitcond.j.not = icmp eq i64 %inc.j, %n
  br i1 %exitcond.j.not, label %for.j.exit, label %for.j.header, !llvm.loop !4

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %exitcond.i.not = icmp eq i64 %inc.i, %m
  br i1 %exitcond.i.not, label %for.i.exit, label %for.i.header, !llvm.loop !5

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

; CHECK-LABEL: @ppep
; CHECK: llvm.loop ![[PPEP_K:[0-9]+]]
; CHECK: llvm.loop ![[PPEP_J:[0-9]+]]
; CHECK: llvm.loop ![[PPEP_I:[0-9]+]]
;
; forall (i ...) {
;   forall (j ...) {
;     forall (k ...) {
;       ...
;     }
;     expr
;   }
; }
;
define void @ppep(i64 %m, i64 %n, i64 %p) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  %cmp.m.not = icmp eq i64 %m, 0
  %cmp.n.not = icmp eq i64 %n, 0
  %cmp.p.not = icmp eq i64 %p, 0
  br i1 %cmp.m.not, label %for.i.exit, label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.j = tail call token @llvm.syncregion.start()
  br i1 %cmp.n.not, label %for.j.exit, label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  %syncreg.k = tail call token @llvm.syncregion.start()
  br i1 %cmp.p.not, label %for.k.exit, label %for.k.header

for.k.header:
  %k = phi i64 [0, %for.j.body ], [ %inc.k, %for.k.latch ]
  detach within %syncreg.k, label %for.k.body, label %for.k.latch

for.k.body:
  reattach within %syncreg.k, label %for.k.latch

for.k.latch:
  %inc.k = add i64 %k, 1
  %exitcond.k.not = icmp eq i64 %inc.k, %p
  br i1 %exitcond.k.not, label %for.k.exit, label %for.k.header, !llvm.loop !6

for.k.exit:
  sync within %syncreg.k, label %for.k.end

for.k.end:
  %0 = add i64 %m, %n
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %exitcond.j.not = icmp eq i64 %inc.j, %n
  br i1 %exitcond.j.not, label %for.j.exit, label %for.j.header, !llvm.loop !7

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %exitcond.i.not = icmp eq i64 %inc.i, %m
  br i1 %exitcond.i.not, label %for.i.exit, label %for.i.header, !llvm.loop !8

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

;-------------------------------------------------------------------------------
;
; CHECK-DAG: ![[TARGET:[0-9]+]] = !{!"tapir.loop.target", i32 32}
; CHECK-DAG: ![[LOWER:[0-9]+]] = !{!"tapir.loop.lowering.enabled", i32 1}
;
; ------------------------------------------------------------------------------
;
; forall (i ...) {
;   expr
;   forall (j ...)
;     ;
; }
;
; CHECK-DAG: ![[PEP_J]] = distinct !{![[PEP_J]], ![[TARGET]], ![[LOWER]]}
; CHECK-DAG: ![[PEP_I]] = distinct !{![[PEP_I]], ![[TARGET]], ![[LOWER]]}
;
;-------------------------------------------------------------------------------
;
; forall (i ...) {
;   expr;
;   forall (j ...) {
;     forall (k ...)
;       ...
;   }
; }
;
; CHECK-DAG: ![[PEPP_K]] = distinct !{![[PEPP_K]], ![[TARGET]], ![[LOWER]]}
; CHECK-DAG: ![[PEPP_J]] = distinct !{![[PEPP_J]], ![[TARGET]], ![[LOWER]]}
; CHECK-DAG: ![[PEPP_I]] = distinct !{![[PEPP_I]], ![[TARGET]], ![[LOWER]]}
;
;-------------------------------------------------------------------------------
;
; forall (i ...) {
;   forall (j ...) {
;     forall (k ...) {
;       ...
;     }
;     expr
;   }
; }
;
; CHECK-DAG: ![[PPEP_K]] = distinct !{![[PPEP_K]], ![[TARGET]], ![[LOWER]]}
; CHECK-DAG: ![[PPEP_J]] = distinct !{![[PPEP_J]], ![[TARGET]], ![[LOWER]]}
; CHECK-DAG: ![[PPEP_I]] = distinct !{![[PPEP_I]], ![[TARGET]], ![[LOWER]]}
;
;-------------------------------------------------------------------------------

!0 = !{!"tapir.loop.target", i32 32}
!1 = distinct !{!1, !0}
!2 = distinct !{!2, !0}
!3 = distinct !{!3, !0}
!4 = distinct !{!4, !0}
!5 = distinct !{!5, !0}
!6 = distinct !{!6, !0}
!7 = distinct !{!7, !0}
!8 = distinct !{!8, !0}

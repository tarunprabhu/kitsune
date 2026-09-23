; The early annotator assumes that nests of tapir loops are perfect and does not
; check anything. Even if it is given obviously imperfectly nested loops, it
; will annotate them with perfect nesting levels. This is the expected behavior
; of the pass, though it is not ideal. The pass expects that the frontend will
; not have let through imperfectly nested loops.
;
; The only exception to this is when a loop at some level has a sibling at the
; same level. In this case, neither loop will be annotated. The rationale for
; this behavior is provided below:
;
;     forall (...) {           forall (...) {
;       expr;                    forall (...)
;       forall (...)               ...
;         ...                    forall (...)
;       expr;                      ...
;     }                        }
;
; Consider the loops above. In principle, we could generate GPU code from the
; loop nest on the left (we specifically mention GPU's because generating GPU
; kernels from loops is more difficult than generating CPU code). On the other
; hand, generating a GPU kernel from the loop nest on the right is extremely
; difficult and will likely never be supported.
;
; RUN: opt -passes="kit-annotate-early" -S %s | FileCheck %s

; CHECK-LABEL: @pep
; CHECK: !llvm.loop ![[PEP_J:[0-9]+]]
; CHECK: !llvm.loop ![[PEP_I:[0-9]+]]
;
; forall (i ...) {
;   expr
;   forall (j ...)
;     ;
; }
;
define void @pep(i64 %m, i64 %n) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.j = tail call token @llvm.syncregion.start()
  %0 = add i64 %m, %n
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
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.j = tail call token @llvm.syncregion.start()
  %0 = add i64 %m, %n
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
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !6

for.k.exit:
  sync within %syncreg.k, label %for.k.end

for.k.end:
  %0 = add i64 %m, %n
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !7

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !8

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

; CHECK-LABEL: @psib
; CHECK: llvm.loop ![[PSIB_J1:[0-9]+]]
; CHECK: llvm.loop ![[PSIB_J2:[0-9]+]]
; CHECK: llvm.loop ![[PSIB_I:[0-9]+]]
;
; forall (i ...) {
;   forall (j1 ...)
;     ...
;   forall (j2 ...)
;     ...
; }
;
define void @psib(i64 %m, i64 %n1, i64 %n2) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.j1 = tail call token @llvm.syncregion.start()
  br label %for.j1.header

for.j1.header:
  %j1 = phi i64 [ 0, %for.i.body ], [ %inc.j1, %for.j1.latch ]
  detach within %syncreg.j1, label %for.j1.body, label %for.j1.latch

for.j1.body:
  reattach within %syncreg.j1, label %for.j1.latch

for.j1.latch:
  %inc.j1 = add i64 %j1, 1
  %cmp.j1 = icmp eq i64 %inc.j1, %n1
  br i1 %cmp.j1, label %for.j1.exit, label %for.j1.header, !llvm.loop !9

for.j1.exit:
  sync within %syncreg.j1, label %for.j1.end

for.j1.end:
  %syncreg.j2 = tail call token @llvm.syncregion.start()
  br label %for.j2.header

for.j2.header:
  %j2 = phi i64 [ 0, %for.j1.end ], [ %inc.j2, %for.j2.latch ]
  detach within %syncreg.j1, label %for.j1.body, label %for.j1.latch

for.j2.body:
  reattach within %syncreg.j2, label %for.j2.latch

for.j2.latch:
  %inc.j2 = add i64 %j2, 1
  %cmp.j2 = icmp eq i64 %inc.j2, %n2
  br i1 %cmp.j2, label %for.j2.exit, label %for.j2.header, !llvm.loop !10

for.j2.exit:
  sync within %syncreg.j2, label %for.i.reattach

for.i.reattach:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !11

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

;-------------------------------------------------------------------------------
;
; CHECK-DAG: ![[D1:[0-9]+]] = !{!"tapir.loop.perfect.depth", i32 1}
; CHECK-DAG: ![[D2:[0-9]+]] = !{!"tapir.loop.perfect.depth", i32 2}
; CHECK-DAG: ![[D3:[0-9]+]] = !{!"tapir.loop.perfect.depth", i32 3}
; CHECK-DAG: ![[L1:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 1}
; CHECK-DAG: ![[L2:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 2}
; CHECK-DAG: ![[L3:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 3}
;
; ------------------------------------------------------------------------------
;
; forall (i ...) {
;   expr;
;   forall (j ...)
;     ;
; }
;
; CHECK-DAG: ![[PEP_J]] = distinct !{![[PEP_J]], {{.+}}, ![[L2]]}
; CHECK-DAG: ![[PEP_I]] = distinct !{![[PEP_I]], {{.+}}, ![[L1]], ![[D2]]}
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
; CHECK-DAG: ![[PEPP_K]] = distinct !{![[PEPP_K]], {{.+}}, ![[L3]]}
; CHECK-DAG: ![[PEPP_J]] = distinct !{![[PEPP_J]], {{.+}}, ![[L2]]}
; CHECK-DAG: ![[PEPP_I]] = distinct !{![[PEPP_I]], {{.+}}, ![[L1]], ![[D3]]}
;
;-------------------------------------------------------------------------------
;
; forall (i ...) {
;   forall (j ...) {
;     forall (k ...) {
;       ...
;     }
;     expr;
;   }
; }
;
; CHECK-DAG: ![[PPEP_K]] = distinct !{![[PPEP_K]], {{.+}}, ![[L3]]}
; CHECK-DAG: ![[PPEP_J]] = distinct !{![[PPEP_J]], {{.+}}, ![[L2]]}
; CHECK-DAG: ![[PPEP_I]] = distinct !{![[PPEP_I]], {{.+}}, ![[L1]], ![[D3]]}
;
;-------------------------------------------------------------------------------
;
; forall (i ...) {
;   forall (j1 ...)
;     ...
;   forall (j2 ...)
;     ...
; }
;
; CHECK-DAG: ![[PSIB_J1]] = distinct !{![[PSIB_J1]], !{{.+}}}
; CHECK-DAG: ![[PSIB_J2]] = distinct !{![[PSIB_J2]], {{.+}}}
; CHECK-DAG: ![[PSIB_I]] = distinct !{![[PSIB_I]], {{.+}}, ![[L1]], ![[D1]]}
;
;-------------------------------------------------------------------------------

!0 = !{!"tapir.loop.target", i32 2}
!1 = distinct !{!1, !0}
!2 = distinct !{!2, !0}
!3 = distinct !{!3, !0}
!4 = distinct !{!4, !0}
!5 = distinct !{!5, !0}
!6 = distinct !{!6, !0}
!7 = distinct !{!7, !0}
!8 = distinct !{!8, !0}
!9 = distinct !{!9, !0}
!10 = distinct !{!10, !0}
!11 = distinct !{!11, !0}

; Check that the correct diagnostics are emitted for each imperfectly nested
; tapir loop in a loop nest where the root has the 'cuda' tapir target.
;
; RUN: opt -passes="kit-verify-prelower" --tapir=cuda -disable-output %s 2>&1 \
; RUN:     | FileCheck %s

; CHECK: parallel loop not perfectly nested
; CHECK-NEXT: from loop 'pepp.loop.j'
; CHECK-NEXT: from function 'pepp'
; CHECK-NEXT: root of tapir loop nest is 'pepp.loop.i'
;
; CHECK: parallel loop not perfectly nested
; CHECK-NEXT: from loop 'pepp.loop.k'
; CHECK-NEXT: from function 'pepp'
; CHECK-NEXT: root of tapir loop nest is 'pepp.loop.i'
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
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !2

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !1

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

; CHECK: parallel loop not perfectly nested
; CHECK-NEXT: from loop 'pppe.loop.k'
; CHECK-NEXT: from function 'pppe'
; CHECK-NEXT: root of tapir loop nest is 'pppe.loop.i'
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
define void @pppe(i64 %m, i64 %n, i64 %p) {
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
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !5

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !4

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

!0 = !{!"tapir.loop.target", i32 2}
!1 = distinct !{!1, !0, !7}
!2 = distinct !{!2, !0, !8}
!3 = distinct !{!3, !0, !9}
!4 = distinct !{!4, !0, !10}
!5 = distinct !{!5, !0, !11}
!6 = distinct !{!6, !0, !12}
!7 = !{!"tapir.loop.name", !"pepp.loop.i"}
!8 = !{!"tapir.loop.name", !"pepp.loop.j"}
!9 = !{!"tapir.loop.name", !"pepp.loop.k"}
!10 = !{!"tapir.loop.name", !"pppe.loop.i"}
!11 = !{!"tapir.loop.name", !"pppe.loop.j"}
!12 = !{!"tapir.loop.name", !"pppe.loop.k"}

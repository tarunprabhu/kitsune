; Check that a simple tapir loop of depth 1 is prepared as expected. This should
; leave the loop unchanged. But the tapir.loop.prepared attribute must be added
; to the loop.
;
; RUN: opt --tapir=serial --passes=kit-prepare -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SYNCREG:.+]] = tail call token @llvm.syncregion.start()
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: %[[IV:.+]] = phi i64
; CHECK-SAME: [ 0, %[[ENTRY]] ]
; CHECK-SAME: [ %[[NEXT:.+]], %[[LATCH:.+]] ]
; CHECK-NEXT: detach within %[[SYNCREG]], label %[[BODY:.+]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: reattach within %[[SYNCREG]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH]]:
; CHECK-NEXT: %[[NEXT:.+]] = add i64 %[[IV]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[NEXT]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[HEADER]],
; CHECK-SAME: !llvm.loop ![[LOOP:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: sync within %[[SYNCREG]], label %[[END:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[END]]:
; CHECK-NEXT: ret void
;
; CHECK-DAG: ![[TARGET:.+]] = !{!"tapir.loop.target", i32 1}
; CHECK-DAG: ![[PREPARED:.+]] = !{!"tapir.loop.prepared"}
; CHECK-DAG: ![[LOOP]] = distinct !{![[LOOP]], ![[TARGET]], ![[PREPARED]]}

define void @f(i64 %n) {
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
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !1

for.i.exit:
  sync within %syncreg, label %for.i.end

for.i.end:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}

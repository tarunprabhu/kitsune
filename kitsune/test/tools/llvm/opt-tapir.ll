; Check that both the --tapir and --tapir-target are valid options for opt.
;
; RUN: opt %s --tapir=serial -passes="tapir-lowering<O2>" -S \
; RUN:     | FileCheck %s
; RUN: opt %s --tapir-target=serial -passes="tapir-lowering<O2>" -S \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @mset
; CHECK: [[ENTRY:.+]]:
; CHECK: [[BODY:.+]]:
; CHECK-NEXT:  %[[IV:.+]] = phi i64 [ %[[INC:.+]], %[[BODY]] ], [ 0, %[[ENTRY]] ]
; CHECK-NEXT:  %[[IDX:.+]] = getelementptr inbounds nuw i64, ptr %{{.}}, i64 %[[IV]]
; CHECK-NEXT:  store i64 %{{.+}}, ptr %[[IDX]]
; CHECK-NEXT:  %[[INC]] = add {{.*}}i64 %[[IV]], 1
; CHECK-NEXT:  %[[COND:.+]] = icmp eq i64 %[[INC]], %{{.+}}
; CHECK-NEXT:  br i1 %[[COND]], label %[[EXIT:.+]], label %[[BODY]]
; CHECK: [[EXIT]]:

target triple = "x86_64-unknown-linux-gnu"

define void @mset(ptr %a, i64 %n, i64 %v) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp4 = icmp sgt i64 %n, 0
  br i1 %cmp4, label %forall.detach, label %forall.sync

forall.detach:
  %i.05 = phi i64 [ %inc, %forall.inc ], [ 0, %entry ]
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %i.05
  store i64 %v, ptr %arrayidx, align 8
  reattach within %syncreg, label %forall.inc

forall.inc:
  %inc = add nuw nsw i64 %i.05, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !0

forall.sync:
  sync within %syncreg, label %forall.end

forall.end:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}

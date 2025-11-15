; REQUIRES: kitsune-examples
;
; Check that a tapir target plugin works as expected on C++ code. We use the
; tapir target plugin demo for consistency with the way LLVM pass plugins are
; tested.
;
; RUN: opt --tapir=custom --tapir-plugin=%kit-ttplugin-demo %s \
; RUN:     -S -o - -O2 \
; RUN:     | FileCheck %s --check-prefix=BOOKEND
;
; BOOKEND: call void @bookend
; BOOKEND-NEXT: call {{.*}}void @mset{{[^(]+}}(
; BOOKEND-NEXT: call void @bookend

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

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.spawn.strategy", i32 4}
!2 = !{!"tapir.loop.target", i32 2048}
!3 = !{!"llvm.loop.unroll.disable"}

; Check that non-constant global variables used in a tapir loop are cloned into
; the kernel module with the correct visibility. The initializer of internal
; non-const globals is "lost" in the process, since their linkage has to be set
; to external. That is ok because the actual value will be copied from host to
; device prior to every launch call.
;
; RUN: opt --tapir=hip -passes='tapir-lowering<O2>,emb-prepare' %s \
; RUN:     | kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @v137 = protected {{.*}}global i32
; CHECK-DAG: @v138 = protected {{.*}}global i32

target triple = "x86_64-unknown-linux-gnu"

@v137 = external global i32, align 4
@v138 = internal global i32 812, align 4

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp4.not = icmp eq i64 %n, 0
  br i1 %cmp4.not, label %forall.sync, label %forall.detach

forall.detach:
  %i.05 = phi i64 [ %inc, %forall.inc ], [ 0, %entry ]
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:
  %0 = load i32, ptr @v137, align 4
  %1 = load i32, ptr @v138, align 4
  %2 = add i32 %0, %1
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %i.05
  store i32 %2, ptr %arrayidx, align 4
  reattach within %syncreg, label %forall.inc

forall.inc:
  %inc = add nuw i64 %i.05, 1
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

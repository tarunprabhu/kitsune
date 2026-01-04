; REQUIRES: kitsune-examples, kitsune-hip
;
; Check that the embedded module passes in a pass plugin is registered with the
; Kitsune pipeline when the kit-lowering meta-pass is used.
;
; NOTE: We only check for the presence of external_func called in the forall
; loop. A second function, the kernel function, will also be present but we
; do not check for it because the name of that function is not guaranteed to
; be consistent.
;
; NOTE: We have separate tests for all tapir targets that use embedded bitcode
; to ensure that, if at least one is built, that this functionality is tested.
; They cannot all be compiled into a single test because that would be treated
; as multi-target compilation, which is not yet supported.
;
; RUN: opt --tapir=hip --tapir-hip-arch=gfx90a \
; RUN:     --tapir-hip-runtime-bcs=%S/input/libdevice.bc \
; RUN:     --load-pass-plugin=%kit-emb-pass-plugin-demo \
; RUN:     -passes='kit-lowering<O1>' -disable-output %s \
; RUN:     | FileCheck %s
;
; CHECK-DAG: external_func
; CHECK-DAG: llvm.amdgcn.workitem.id.x

target triple = "x86_64-unknown-linux-gnu"

declare i64 @external_func(i64)

define void @mset(ptr %a, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp4 = icmp sgt i64 %n, 0
  br i1 %cmp4, label %forall.detach, label %forall.sync

forall.detach:
  %i.05 = phi i64 [ %inc, %forall.inc ], [ 0, %entry ]
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:
  %v = call i64 @external_func(i64 %i.05)
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
!1 = !{!"tapir.loop.spawn.strategy", i32 3}
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"llvm.loop.unroll.disable"}

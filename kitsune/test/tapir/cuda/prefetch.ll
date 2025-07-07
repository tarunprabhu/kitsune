; Check that the --tapir-gpu-prefetch option is handled correctly
;
; -----------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda -passes='tapir-lowering<O2>' -S %s \
; RUN:     --tapir-gpu-prefetch=true \
; RUN:     | FileCheck %s -check-prefix PREFETCH
;
; PREFETCH: define {{.+}} @f
; PREFETCH: call {{.+}} @llvm.kit.async.prefetch.htod(i32 2,
; PREFETCH: call {{.+}} @llvm.kit.async.launch.kernel(i32 2,
; PREFETCH: ret void
; PREFETCH-NEXT: }
;
; -----------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda -passes='tapir-lowering<O2>' -S %s \
; RUN:     --tapir-gpu-prefetch=false \
; RUN:     | FileCheck %s -check-prefix NO-PREFETCH
;
; NO-PREFETCH: define {{.+}} @f
; NO-PREFETCH-NOT: call {{.+}} @llvm.kit.async.prefetch.htod(i32 2,
; NO-PREFETCH: call {{.+}} @llvm.kit.async.launch.kernel(i32 2,
; NO-PREFETCH: ret void
; NO-PREFETCH-NEXT: }
;
; -----------------------------------------------------------------------------

target triple = "x86_64-unknown-linux-gnu"

define void @f(ptr %c, i32 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp5 = icmp sgt i32 %n, 0
  br i1 %cmp5, label %forall.detach.preheader, label %forall.sync

forall.detach.preheader:                          ; preds = %entry
  %wide.trip.count = zext nneg i32 %n to i64
  br label %forall.detach

forall.detach:                                    ; preds = %forall.detach.preheader, %forall.inc
  %indvars.iv = phi i64 [ 0, %forall.detach.preheader ], [ %indvars.iv.next, %forall.inc ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:                                      ; preds = %forall.detach
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  store i32 %n, ptr %arrayidx, align 4
  reattach within %syncreg, label %forall.inc

forall.inc:                                       ; preds = %forall.body, %forall.detach
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !0

forall.sync:                                      ; preds = %forall.inc, %entry
  sync within %syncreg, label %forall.end

forall.end:                                       ; preds = %forall.sync
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}

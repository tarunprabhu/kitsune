; Check that functions with the __nv and __nv_fast prefixes are left as is.
;
; RUN: opt --tapir=cuda --tapir-cuda-runtime-bc=%S/input/libdevice.ll %s \
; RUN:     -passes='tapir-lowering<O2>,emb-resolve-libdevice-calls' \
; RUN:     | %kitmbc -S \
; RUN:     | FileCheck %s
;
; CHECK: %[[ASF:.+]] = uitofp nneg i64
; CHECK-NEXT: %[[SIN:.+]] = tail call float @__nv_sinf(float %[[ASF]])
; CHECK-NEXT: %[[SINF:.+]] = tail call float @__nv_fast_sinf(float %[[SIN]])
; CHECK-NEXT: store float %[[SINF]],

target triple = "x86_64-pc-linux-gnu"

declare float @__nv_sinf(float) #2
declare float @__nv_fast_sinf(float) #2

; Function Attrs: nounwind memory(argmem: write) uwtable
define dso_local void @f(ptr nocapture noundef writeonly %c, i64 noundef %n) #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp5 = icmp sgt i64 %n, 0
  br i1 %cmp5, label %forall.detach.preheader, label %forall.sync

forall.detach.preheader:                          ; preds = %entry
  br label %forall.detach

forall.detach:                                    ; preds = %forall.detach.preheader, %forall.inc
  %indvars.iv = phi i64 [ 0, %forall.detach.preheader ], [ %indvars.iv.next, %forall.inc ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:                                      ; preds = %forall.detach
  %arrayidx = getelementptr inbounds float, ptr %c, i64 %indvars.iv
  %asf = sitofp i64 %indvars.iv to float
  %sin = tail call float @__nv_sinf(float %asf)
  %sinf = tail call float @__nv_fast_sinf(float %sin)
  store float %sinf, ptr %arrayidx, align 4
  reattach within %syncreg, label %forall.inc

forall.inc:                                       ; preds = %forall.body, %forall.detach
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !0

forall.sync:                                      ; preds = %forall.inc, %entry
  sync within %syncreg, label %forall.end

forall.end:                                       ; preds = %forall.sync
  ret void
}

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare token @llvm.syncregion.start() #1

attributes #0 = { nounwind memory(argmem: write) uwtable }
attributes #1 = { mustprogress nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}

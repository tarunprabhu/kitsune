; Check that the loop bounds of the tapir loop are correctly replaced in the
; outlined kernel function when a 1D kernel is launched.
;
; NOTE: Currently, we use a grainsize of 1 i.e. every thread computes a single
; iteration of the tapir loop, but if that changes, this test must be updated.
;
; RUN: opt --tapir=hip --tapir-hip-arch="gfx90a" %s \
; RUN:     -passes='tapir-lowering<O2>' \
; RUN:     | kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: define {{.+}}(i64 {{.*}}%[[UB:[^,]+]], i64 {{[^,]+}}, i64 {{[^,]+}}, ptr {{.*}}%[[BUF:[^,]+]], i64 {{.*}}%[[N:[^)]+]]) #[[ATTRS:[0-9]+]]
; CHECK-NEXT: [[PREHEADER:.+]]:
; CHECK: %[[WITEM:.+]] = {{.*}}call i32 @llvm.amdgcn.workitem.id.x()
; CHECK: %[[TID:.+]] = zext i32 %[[WITEM]] to i64
; CHECK: %[[BDIM:.+]] = {{.*}}call i64 @__ockl_get_local_size(i32 0)
; CHECK: %[[WGRP:.+]] = {{.*}}call i32 @llvm.amdgcn.workgroup.id.x()
; CHECK: %[[BIDX:.+]] = zext i32 %[[WGRP]] to i64
; CHECK: %[[BOFF:.+]] = mul i64 %[[BIDX]], %[[BDIM]]
; CHECK: %[[TIV:.+]] = add i64 %[[TID]], %[[BOFF]]
; CHECK: %[[TEND:.+]] = add i64 %[[TIV]], 1
;
; CHECK: [[HEADER:.+]]:
; CHECK: %[[IV:.+]] = phi i64 [ %[[IV_NEXT:.+]], %[[LATCH:.+]] ], [ %[[TIV]], %[[PREHEADER]] ]
; CHECK: %[[IV_NEXT:.+]] = add {{.*}}i64 %[[IV]], 1
; CHECK: %[[COND:.+]] = icmp eq i64 %[[IV_NEXT]], %[[TEND]]
; CHECK: br i1 %[[COND]], label %[[BBEXIT:.+]], label %[[HEADER]]
;
; CHECK: [[BBEXIT]]:
; CHECK-NEXT: ret void
;
; CHECK: attributes #[[ATTRS]] = {
; CHECK-SAME: kit_kernel

target triple = "x86_64-pc-linux-gnu"

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
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  store i64 %n, ptr %arrayidx, align 4
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

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}

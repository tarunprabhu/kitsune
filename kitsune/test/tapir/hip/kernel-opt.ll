; Check that when the standard sequence of optimization passes are run on the
; code, the results as expected.
;
; ------------------------------------------------------------------------------
;
; We have to set the optimization level of tapir lowering to non-zero for it to
; work. However, we can override the optimization level used on the embedded
; bitcode module. The lowering only adjusts the bounds of the original tapir
; loop. Setting the optimization level to O0 retains this loop.
;
; RUN: opt --tapir=hip --tapir-hip-runtime-bcs=%S/input/libdevice.ll %s \
; RUN:     -passes='tapir-lowering<O1>,emb-optimize' -emb-opt-level=0 \
; RUN:     | %kitmbc -S \
; RUN:     | FileCheck %s --check-prefix=O0
;
; O0: define {{.+}} @__kithip_{{.+}}(i64
; O0: = phi i64
;
; ------------------------------------------------------------------------------
;
; NOTE: This assumes that the grainsize is 1. For now, this is hard-coded into
; the hip tapir target. If we ever allow this to be configurable, and set the
; default value to something other than 1, this needs to be changed.
;
; If compiling with optimizations, the loop will be removed since the trip
; is determined to be 1. If the grain size is changed to be greater than 1, we
; may need to check for unrolling.
;
; RUN: opt --tapir=hip --tapir-hip-runtime-bcs=%S/input/libdevice.ll %s \
; RUN:     -passes='tapir-lowering<O2>,emb-optimize' \
; RUN:     | %kitmbc -S \
; RUN:     | FileCheck %s --check-prefix=O2
;
; O2-NOT: = phi i64
; O2: define {{.+}} @__kithip_{{.+}}(i64 {{.*}}%[[UB:[^,]+]], i64 {{[^,]+}}, i64 {{[^,]+}}, ptr {{.*}}%[[BUF:[^,]+]], i64 {{.*}}%[[N:[^)]+]]) {{.*}}#[[ATTRS:[0-9]+]]
; O2-NEXT: [[BBENTRY:.+]]:
; O2-NEXT: %[[WITEM:.+]] = {{.*}}call i32 @llvm.amdgcn.workitem.id.x()
; O2-NEXT: %[[TID:.+]] = zext i32 %[[WITEM]] to i64
; O2-NEXT: %[[BDIM:.+]] = {{.*}}call i64 @__ockl_get_local_size(i32 0)
; O2-NEXT: %[[WGRP:.+]] = {{.*}}call i32 @llvm.amdgcn.workgroup.id.x()
; O2-NEXT: %[[BIDX:.+]] = zext i32 %[[WGRP]] to i64
; O2-NEXT: %[[BOFF:.+]] = mul i64 %[[BDIM]], %[[BIDX]]
; O2-NEXT: %[[TIV:.+]] = add i64 %[[BOFF]], %[[TID]]
; O2-NEXT: %[[COND:.+]] = icmp ult i64 %[[TIV]], %[[UB]]
; O2-NEXT: br i1 %[[COND]], label %[[BBBODY:[^,]+]], label %[[BBEXIT:.+]]
; O2: [[BBBODY]]:
; O2-NEXT: %[[BUFCST:.+]] = addrspacecast ptr %[[BUF]] to ptr addrspace(1)
; O2-NEXT: %[[ARRIDX:.+]] = getelementptr {{.+}}, ptr {{.*}}%[[BUFCST]], i64 %[[TIV]]
; O2-NEXT: store i64 %[[N]], ptr {{.*}}%[[ARRIDX]]
; O2-NEXT: br label %[[BBEXIT]]
; O2: [[BBEXIT]]:
; O2-NEXT: ret void
;
; O2: attributes #[[ATTRS]] = {
; O2-SAME: kit_kernel
;
; ------------------------------------------------------------------------------

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

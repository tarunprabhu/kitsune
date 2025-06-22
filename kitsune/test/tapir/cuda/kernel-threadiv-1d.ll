; Check that the induction variable for a 1D launch is computed correctly and
; bypasses the body of the loop if out of bounds
;
; RUN: opt --tapir=cuda --tapir-cuda-arch="sm_72" %s \
; RUN:     -passes='tapir-lowering<O2>' \
; RUN:     | %kitmbc -S \
; RUN:     | FileCheck %s
;
; CHECK: define {{.+}}(i64 {{[^%]*}}%[[UB:[^,]+]], {{.+}}) #[[ATTRS:[0-9]+]]
; CHECK: %[[TID:.+]] = {{.*}}call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK: %[[BIDX:.+]] = {{.*}}call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
; CHECK: %[[BDIM:.+]] = {{.*}}call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
; CHECK: %[[BOFF:.+]] = mul i32 %[[BIDX]], %[[BDIM]]
; CHECK: %[[IV32:.+]] = add i32 %[[TID]], %[[BOFF]]
; CHECK: %[[TIV:.+]] = zext i32 %[[IV32]] to i64
; CHECK: %[[COND:.+]] = icmp uge i64 %[[TIV]], %[[UB]]
; CHECK-NEXT: br i1 %[[COND]], label %[[BBEXIT:[^,]+]],
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

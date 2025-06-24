; Check that alloca instructions have the correct address space annotation
; added to them.
;
; RUN: opt --tapir=hip -passes='loop-spawning,emb-prepare' %s \
; RUN:     | %kitmbc -S \
; RUN:     | FileCheck %s
;
; CHECK: %[[A1:.+]] = alloca i64, align 8, addrspace(5)
; CHECK: %[[A2:.+]] = alloca i64, align 8, addrspace(5)
; CHECK: %[[CST1:.+]] = addrspacecast ptr addrspace(5) %[[A1]] to ptr
; CHECK: %[[CST2:.+]] = addrspacecast ptr addrspace(5) %[[A2]] to ptr
; CHECK: %[[V0:.+]] = ptrtoint ptr %[[CST1]] to i64
; CHECK: %[[V1:.+]] = ptrtoint ptr %[[CST2]] to i64

target triple = "x86_64-unknown-linux-gnu"

%"class.kitsune::mobile_ptr" = type { ptr addrspace(67) }

@stash = global ptr zeroinitializer

; Function Attrs: mustprogress norecurse nounwind memory(argmem: readwrite) uwtable
define fastcc void @_ZL20initialize_variablesiN7kitsune10mobile_ptrIfEES1_(i64 %nelr, ptr %variables, ptr %ff_variable) #1 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp14 = icmp sgt i64 %nelr, 0
  br i1 %cmp14, label %forall.detach.preheader, label %forall.sync

forall.detach.preheader:                          ; preds = %entry
  %invariant.gep = getelementptr inbounds nuw float, ptr %variables, i64 %nelr
  br label %forall.detach

forall.detach:                                    ; preds = %forall.inc, %forall.detach.preheader
  %indvars.iv = phi i64 [ 0, %forall.detach.preheader ], [ %indvars.iv.next, %forall.inc ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:                                      ; preds = %forall.detach
  %a1 = alloca i64, align 8
  %a2 = alloca i64, align 8
  %0 = ptrtoint ptr %a1 to i64
  %1 = ptrtoint ptr %a2 to i64
  %2 = add i64 %0, %1
  %3 = add i64 %2, %indvars.iv
  %4 = load float, ptr %ff_variable, align 4
  %arrayidx.i = getelementptr inbounds nuw float, ptr %variables, i64 %3
  store float %4, ptr %arrayidx.i, align 4
  reattach within %syncreg, label %forall.inc

forall.inc:                                       ; preds = %forall.body, %forall.detach
  %exitcond.not = icmp eq i64 %indvars.iv.next, %nelr
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !0

forall.sync:                                      ; preds = %forall.inc, %entry
  sync within %syncreg, label %forall.end

forall.end:                                       ; preds = %forall.sync
  ret void
}

attributes #0 = { mustprogress norecurse nounwind uwtable memory(argmem: readwrite) "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}
!3 = distinct !{!3, !2, !4}
!4 = !{!"llvm.loop.mustprogress"}

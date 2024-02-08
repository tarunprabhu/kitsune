; Check that both the --tapir and --tapir-target are valid options for opt.
;
; RUN: opt < %s --tapir=serial -passes="tapir-lowering<O2>" -S | FileCheck %s
; RUN: opt < %s --tapir-target=serial -passes="tapir-lowering<O2>" -S | FileCheck %s

; CHECK-LABEL: mset
; CHECK: [[ENTRY:.+]]:
; CHECK: [[BODY:.+]]:
; CHECK-NEXT:  %[[IV:.+]] = phi i64 [ %[[INC:.+]], %[[BODY]] ], [ 0, %[[ENTRY]] ]
; CHECK-NEXT:  %[[IDX:.+]] = getelementptr inbounds nuw i64, ptr %{{.}}, i64 %[[IV]]
; CHECK-NEXT:  store i64 %{{.+}}, ptr %[[IDX]]
; CHECK-NEXT:  %[[INC]] = add nuw nsw i64 %[[IV]], 1
; CHECK-NEXT:  %[[COND:.+]] = icmp eq i64 %[[INC]], %{{.+}}
; CHECK-NEXT:  br i1 %[[COND]], label %[[EXIT:.+]], label %[[BODY]]
; CHECK: [[EXIT]]:

; ModuleID = '-'
source_filename = "-"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind memory(argmem: write) uwtable
define dso_local void @mset(ptr nocapture noundef writeonly %a, i64 noundef %n, i64 noundef %v) local_unnamed_addr #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp4 = icmp sgt i64 %n, 0
  br i1 %cmp4, label %forall.detach, label %forall.sync

forall.detach:                                    ; preds = %entry, %forall.inc
  %i.05 = phi i64 [ %inc, %forall.inc ], [ 0, %entry ]
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:                                      ; preds = %forall.detach
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %i.05
  store i64 %v, ptr %arrayidx, align 8
  reattach within %syncreg, label %forall.inc

forall.inc:                                       ; preds = %forall.body, %forall.detach
  %inc = add nuw nsw i64 %i.05, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !4

forall.sync:                                      ; preds = %forall.inc, %entry
  sync within %syncreg, label %forall.end

forall.end:                                       ; preds = %forall.sync
  ret void
}

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare token @llvm.syncregion.start() #1

attributes #0 = { nounwind memory(argmem: write) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = distinct !{!4, !5, !6, !7}
!5 = !{!"tapir.loop.spawn.strategy", i32 0}
!6 = !{!"tapir.loop.target", i32 2}
!7 = !{!"llvm.loop.unroll.disable"}

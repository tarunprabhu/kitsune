; Check that the command line options make it to the options objects
;
; RUN: opt --tapir=cuda -passes="tapir-lowering<O2>" -o /dev/null %s \
; RUN:      --tapir-verbose 2>&1 \
; RUN:      | FileCheck %s -check-prefixes ALL,DEFAULT
;
; RUN: opt --tapir=cuda -passes="tapir-lowering<O2>" -o /dev/null %s \
; RUN:      --tapir-verbose --kitrt-verbose 2>&1\
; RUN:      | FileCheck %s -check-prefixes ALL,RUNTIME
;
; RUN: opt --tapir=cuda -passes="tapir-lowering<O2>" -o /dev/null %s \
; RUN:      --tapir-verbose --tapir-cuda-arch=sm_72 2>&1\
; RUN:      | FileCheck %s -check-prefixes ALL,ARCH
;
; RUN: opt --tapir=cuda -passes="tapir-lowering<O2>" -o /dev/null %s \
; RUN:      --tapir-verbose --tapir-threads-per-block=64 2>&1\
; RUN:      | FileCheck %s -check-prefixes ALL,TPB
;
; RUN: opt --tapir=cuda -passes="tapir-lowering<O2>" -o /dev/null %s \
; RUN:      --tapir-verbose --tapir-max-threads-per-block=128 2>&1\
; RUN:      | FileCheck %s -check-prefixes ALL,MTPB
;
; ALL: 'cuda' tapir target options
; DEFAULT:   Runtime verbose: 1
; RUNTIME:   Runtime verbose: 1
; OPTLEVEL:  Optimization level: O2
; ARCH:      GPU arch: sm_72
; TPB:       Fixed threads/block: 64
; MTPB:      Max threads/block: 128

; ModuleID = 'clopts.c'
source_filename = "clopts.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind memory(argmem: write) uwtable
define dso_local void @f(ptr nocapture noundef writeonly %c, i32 noundef %n) local_unnamed_addr #0 {
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
  store i32 %n, ptr %arrayidx, align 4, !tbaa !5
  reattach within %syncreg, label %forall.inc

forall.inc:                                       ; preds = %forall.body, %forall.detach
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !9

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
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 19.1.2 (git@github.com:tarunprabhu/kitsune.git 0ab68f142927b9548ac0bc51a82f9bf5e859b384)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = distinct !{!9, !10, !11}
!10 = !{!"tapir.loop.spawn.strategy", i32 1}
!11 = !{!"llvm.loop.unroll.disable"}

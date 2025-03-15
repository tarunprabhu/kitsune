; Check that the global ctor calls the appropriate functions in Kitsune's
; runtime depending on the command line arguments passed.
;
; RUN: opt --tapir=cuda -S -passes='tapir-lowering<O2>' -S %s 2>&1 \
; RUN:     | FileCheck %s -check-prefix DEFAULT
;
; Currently, if a max-threads-per-block option is not used, the CudaABI
; nevertheless sets the max to 1024.
;
; DEFAULT: @llvm.global_ctors = appending {{.+}}, ptr @kitcu.ctor{{[^ ]+}},
; DEFAULT: define {{.+}} @kitcu.ctor{{.*}}
; DEFAULT: call {{.+}}__kitcuda_initialize()
; DEFAULT-NOT: call {{.+}}__kitcuda_set_default_threads_per_blk
; DEFAULT: call {{.+}}__kitcuda_set_max_threads_per_blk(i32 1024)
; DEFAULT-NOT: call {{.+}}__kitrt_enable_verbose_mode()
; DEFAULT-DAG: call {{.+}}__kitcuda_enable_launch_refinement(i8 1)
; DEFAULT-DAG: call {{.+}}__cudaRegisterFatBinary
; DEFAULT: call {{.+}}__cudaRegisterFatBinaryEnd
; DEFAULT: call {{.+}}atexit(ptr nonnull @kitcu.dtor{{[^ ]*}})
; DEFAULT: }
;
; DEFAULT: define {{.+}} @kitcu.dtor{{.*}}
; DEFAULT: call {{.+}} @__cudaUnregisterFatBinary
; DEFAULT: call {{.+}} @__kitcuda_destroy
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda -S -passes='tapir-lowering<O2>' -S %s \
; RUN:     --tapir-threads-per-block=77 \
; RUN:     | FileCheck %s -check-prefix TPB
;
; TPB-LABEL: kitcu.ctor{{.*}}
; TPB: call {{.+}}__kitcuda_set_default_threads_per_blk(i32 77)
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda -S -passes='tapir-lowering<O2>' -S %s \
; RUN:     --tapir-max-threads-per-block=29 \
; RUN:     | FileCheck %s -check-prefix MTPB
;
; MTPB-LABEL: kitcu.ctor{{.*}}
; MTPB: call {{.+}}__kitcuda_set_max_threads_per_blk(i32 29)
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda -S -passes='tapir-lowering<O2>' -S %s \
; RUN:     --tapir-verbose \
; RUN:     | FileCheck %s -check-prefix VERBOSE
;
; RUN: opt --tapir=cuda -S -passes='tapir-lowering<O2>' -S %s \
; RUN:     --kitrt-verbose \
; RUN:     | FileCheck %s -check-prefix VERBOSE
;
; VERBOSE-LABEL: kitcu.ctor{{.*}}
; VERBOSE: call {{.+}}__kitrt_enable_verbose_mode()
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda -S -passes='tapir-lowering<O2>' -S %s \
; RUN:     -cuabi-refine-launches=false \
; RUN:     | FileCheck %s -check-prefix NOREFINE
;
; NOREFINE-LABEL: kitcu.ctor{{.*}}
; NOREFINE: call {{.+}}__kitcuda_enable_launch_refinement(i8 0)
;
; ----------------------------------------------------------------------------

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

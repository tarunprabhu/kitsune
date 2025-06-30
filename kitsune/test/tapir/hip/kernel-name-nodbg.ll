; Check that the names of the outlined kernel functions are as expected. This
; contains both mangled and demangled function names. This checks that the names
; are demangled when generating the outlined kernel name.
;
; NOTE: At this time, the generated name is obtained from the source file and
; debug info, if available. The approach currently used still runs (low) risk of
; collisions with other function names. Eventually, we will switch to some form
; of name mangling to eliminate the change of collisions. When that happens,
; this test may need to be updated/removed.
;
; RUN: opt --tapir=hip %s \
; RUN:     -passes='tapir-lowering<O2>' \
; RUN:     | kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-DAG: define {{.+}} @__kithip_loop_scale_0(
; CHECK-DAG: define {{.+}} @__kithip_loop_xlate_1(
; CHECK-DAG: define {{.+}} @__kithip_loop_xlate_2(

target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress nounwind memory(argmem: readwrite) uwtable
define dso_local void @_Z5scalePffm(ptr nocapture noundef %buf, float noundef %factor, i64 noundef %n) local_unnamed_addr #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp4.not = icmp eq i64 %n, 0
  br i1 %cmp4.not, label %forall.sync, label %forall.detach

forall.detach:                                    ; preds = %entry, %forall.inc
  %i.05 = phi i64 [ %inc, %forall.inc ], [ 0, %entry ]
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:                                      ; preds = %forall.detach
  %arrayidx = getelementptr inbounds nuw float, ptr %buf, i64 %i.05
  %0 = load float, ptr %arrayidx, align 4
  %mul = fmul float %factor, %0
  store float %mul, ptr %arrayidx, align 4
  reattach within %syncreg, label %forall.inc

forall.inc:                                       ; preds = %forall.body, %forall.detach
  %inc = add nuw i64 %i.05, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !0

forall.sync:                                      ; preds = %forall.inc, %entry
  sync within %syncreg, label %forall.end

forall.end:                                       ; preds = %forall.sync
  ret void
}

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare token @llvm.syncregion.start() #1

; Function Attrs: mustprogress nounwind memory(argmem: readwrite) uwtable
define dso_local void @xlate(ptr nocapture noundef %buf, float noundef %dist, i64 noundef %n) local_unnamed_addr #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp4.not = icmp eq i64 %n, 0
  br i1 %cmp4.not, label %forall.sync, label %forall.detach

forall.detach:                                    ; preds = %entry, %forall.inc
  %i.05 = phi i64 [ %inc, %forall.inc ], [ 0, %entry ]
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:                                      ; preds = %forall.detach
  %arrayidx = getelementptr inbounds nuw float, ptr %buf, i64 %i.05
  %0 = load float, ptr %arrayidx, align 4
  %add = fadd float %dist, %0
  store float %add, ptr %arrayidx, align 4
  reattach within %syncreg, label %forall.inc

forall.inc:                                       ; preds = %forall.body, %forall.detach
  %inc = add nuw i64 %i.05, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !3

forall.sync:                                      ; preds = %forall.inc, %entry
  sync within %syncreg, label %forall.ph.2

forall.ph.2:
  %syncreg.2 = tail call token @llvm.syncregion.start()
  %cmp4.not.2 = icmp eq i64 %n, 0
  br i1 %cmp4.not.2, label %forall.sync.2, label %forall.detach.2

forall.detach.2:
  %i.06 = phi i64 [ %inc.2, %forall.inc.2 ], [ 0, %forall.ph.2 ]
  detach within %syncreg.2, label %forall.body.2, label %forall.inc.2

forall.body.2:
  %arrayidx.2 = getelementptr inbounds nuw float, ptr %buf, i64 %i.06
  %1 = load float, ptr %arrayidx.2, align 4
  %add.2 = fadd float %dist, %1
  store float %add.2, ptr %arrayidx.2, align 4
  reattach within %syncreg.2, label %forall.inc.2

forall.inc.2:
  %inc.2 = add nuw i64 %i.06, 1
  %exitcond.not.2 = icmp eq i64 %inc.2, %n
  br i1 %exitcond.not.2, label %forall.sync.2, label %forall.detach.2, !llvm.loop !4

forall.sync.2:
  sync within %syncreg.2, label %forall.end

forall.end:
  ret void
}

; Function Attrs: nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
declare ptr @llvm.kitrt.launch.kernel(i32 immarg, ptr, ptr, ptr, i64, i32, ptr, ptr) #2

attributes #0 = { mustprogress nounwind memory(argmem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) }

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}
!3 = distinct !{!3, !1, !2}
!4 = distinct !{!4, !1, !2}

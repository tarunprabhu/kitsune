; Check that the command line option to force inline all device functions
; (including those that have the noinline attribute) is handled correctly.
;
; RUN: opt --tapir=hip --tapir-hip-arch=gfx90a %s \
; RUN:     -passes='tapir-lowering<O2>,prepare-emb-bc' -emb-inline-all-force \
; RUN:     | %kitmbc -S \
; RUN:     | FileCheck %s
;
; CHECK: define {{.+}} @device_func{{.+}} #[[ATTRS:[0-9]+]]
; CHECK: attributes #[[ATTRS]] = {
; CHECK-SAME: alwaysinline
; CHECK-NOT: noinline

target triple = "x86_64-pc-linux-gnu"

; Function Attrs: memory(argmem: none) noinline willreturn
define i64 @device_func(i64 noundef %n) #2 {
entry:
  ret i64 %n
}

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
  %.call = call i64 @device_func(i64 %n)
  store i64 %.call, ptr %arrayidx, align 4
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
attributes #2 = { memory(argmem: none) noinline willreturn "target-cpu"="x86-64" "target-features"="+sse4.1" "tune-cpu"="x86-64-v2" }

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}

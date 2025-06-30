; Check that non-constant global variables have the correct address space
; annotation.
;
; RUN: opt --tapir=hip -passes='tapir-lowering<O2>' %s \
; RUN:     | kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @v137 = {{.*}}addrspace(1) global i32

target triple = "x86_64-unknown-linux-gnu"

@v137 = external local_unnamed_addr global i32, align 4

define dso_local void @f(ptr nocapture noundef writeonly %c, i64 noundef %n) #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp4.not = icmp eq i64 %n, 0
  br i1 %cmp4.not, label %forall.sync, label %forall.detach

forall.detach:                                    ; preds = %entry, %forall.inc
  %i.05 = phi i64 [ %inc, %forall.inc ], [ 0, %entry ]
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:                                      ; preds = %forall.detach
  %0 = load i32, ptr @v137, align 4
  %arrayidx = getelementptr inbounds nuw i32, ptr %c, i64 %i.05
  store i32 %0, ptr %arrayidx, align 4
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

attributes #0 = { mustprogress nounwind memory(read, argmem: write, inaccessiblemem: none) uwtable }
attributes #1 = { mustprogress nounwind willreturn memory(argmem: readwrite) }

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}

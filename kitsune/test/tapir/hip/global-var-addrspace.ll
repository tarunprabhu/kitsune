; Check that any non-constant global variables are in the correct address space
; in the kernel module
;
; RUN: opt --tapir=hip -passes='loop-spawning' %s \
; RUN:     | %kitmbc -S \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @v137 = {{[A-Za-z_]+}} addrspace(1) global i32
; CHECK-DAG: @v138 = {{[A-Za-z_]+}} addrspace(1) global i32
; CHECK-DAG: @v139 = {{[A-Za-z_]+}} addrspace(1) global i32

target triple = "x86_64-unknown-linux-gnu"

@v137 = global i32 13, align 4
@v138 = external global i32, align 4
@v139 = internal global i32 291, align 4

define dso_local void @f(ptr nocapture noundef writeonly %c, i64 noundef %n) #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp4.not = icmp eq i64 %n, 0
  br i1 %cmp4.not, label %forall.sync, label %forall.ph

forall.ph:
  br label %forall.detach

forall.detach:                                    ; preds = %entry, %forall.inc
  %i.05 = phi i64 [ %inc, %forall.inc ], [ 0, %forall.ph ]
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:                                      ; preds = %forall.detach
  %0 = load i32, ptr @v137, align 4
  %1 = load i32, ptr @v138, align 4
  %2 = add nsw i32 %0, %1
  %3 = load i32, ptr @v139, align 4
  %4 = add nsw i32 %2, %3
  %arrayidx = getelementptr inbounds nuw i32, ptr %c, i64 %i.05
  store i32 %4, ptr %arrayidx, align 4
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

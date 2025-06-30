; Check that if the same global is used in two separate tapir loops, only a
; single instance of the global is created in the kernel module.
;
; RUN: opt --tapir=hip -passes='tapir-lowering<O2>' %s \
; RUN:     | kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @v137{{[^ ]*}} = protected addrspace(1) global i32
; CHECK-DAG: @v138{{[^ ]*}} = protected addrspace(1) global i32
; CHECK-NOT: @v137{{.*}} =
; CHECK-NOT: @v138{{.*}} =

target triple = "x86_64-unknown-linux-gnu"

@v137 = external local_unnamed_addr global i32, align 4
@v138 = internal global i32 zeroinitializer, align 4

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
  %1 = load i32, ptr @v138, align 4
  %2 = add i32 %0, %1
  %arrayidx = getelementptr inbounds nuw i32, ptr %c, i64 %i.05
  store i32 %2, ptr %arrayidx, align 4
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

define dso_local void @g(ptr nocapture noundef writeonly %c, i64 noundef %n) #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp4.not = icmp eq i64 %n, 0
  br i1 %cmp4.not, label %forall.sync, label %forall.detach

forall.detach:                                    ; preds = %entry, %forall.inc
  %i.05 = phi i64 [ %inc, %forall.inc ], [ 0, %entry ]
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:                                      ; preds = %forall.detach
  %0 = load i32, ptr @v137, align 4
  %1 = load i32, ptr @v138, align 4
  %2 = sub i32 %0, %1
  %arrayidx = getelementptr inbounds nuw i32, ptr %c, i64 %i.05
  store i32 %2, ptr %arrayidx, align 4
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

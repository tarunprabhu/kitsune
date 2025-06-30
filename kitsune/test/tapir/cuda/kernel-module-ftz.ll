; Check that the ftz flag can be overridden in the kernel module annotations.
; TODO: Should also add a check that this has the expected effect on the kernel
; module as well.
;
; RUN: opt %s --tapir=cuda -passes='tapir-lowering<O2>' \
; RUN:     | kit-mbc -S \
; RUN:     | FileCheck %s -check-prefixes=ALL,DEFAULT
;
; RUN: opt %s --tapir=cuda -passes='tapir-lowering<O2>' -cuabi-ftz \
; RUN:     | kit-mbc -S \
; RUN:     | FileCheck %s -check-prefixes=ALL,FTZ
;
; ALL: !llvm.module.flags = !{{{.*}}![[FTZ:[0-9]+]]{{.*}}}
;
; DEFAULT: ![[FTZ]] = !{i32 4, !"nvvm-reflect-ftz", i32 0}
; FTZ: ![[FTZ]] = !{i32 4, !"nvvm-reflect-ftz", i32 1}

target triple = "x86_64-pc-linux-gnu"

; Function Attrs: nounwind memory(argmem: write) uwtable
define dso_local void @f1(ptr nocapture noundef writeonly %c, i32 noundef %n) #0 {
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
  store i32 %n, ptr %arrayidx, align 4
  reattach within %syncreg, label %forall.inc

forall.inc:                                       ; preds = %forall.body, %forall.detach
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
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

; Check that the tapir target copies the llvm ident metadata into the device
; module.
;
; RUN: opt %s --tapir=cuda -passes='tapir-lowering<O2>' \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: !llvm.ident = !{![[IDENT:[0-9]+]]}
;
; CHECK: ![[IDENT]] = !{!"clang 67.3"}

target triple = "x86_64-pc-linux-gnu"

define void @f1(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp5 = icmp sgt i64 %n, 0
  br i1 %cmp5, label %preheader, label %forall.sync

preheader:
  br label %forall.detach

forall.detach:
  %indvars.iv = phi i64 [ 0, %preheader ], [ %indvars.iv.next, %forall.inc ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  store i64 %n, ptr %arrayidx, align 4
  reattach within %syncreg, label %forall.inc

forall.inc:
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !0

forall.sync:
  sync within %syncreg, label %forall.end

forall.end:
  ret void
}

!llvm.ident = !{!3}

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 2}
!2 = !{!"llvm.loop.unroll.disable"}
!3 = !{!"clang 67.3"}

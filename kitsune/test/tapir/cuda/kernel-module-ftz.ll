; Check that the ftz flag can be overridden in the kernel module annotations.
; TODO: Should also add a check that this has the expected effect on the kernel
; module as well.
;
; RUN: opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:     --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:     -passes='tapir-lowering<O2>' -S %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s -check-prefixes=ALL,DEFAULT
;
; RUN: opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:     --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:     -cuabi-ftz \
; RUN:     -passes='tapir-lowering<O2>' -S %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s -check-prefixes=ALL,FTZ
;
; ALL: !llvm.module.flags = !{{{.*}}![[FTZ:[0-9]+]]{{.*}}}
;
; DEFAULT: ![[FTZ]] = !{i32 4, !"nvvm-reflect-ftz", i32 0}
; FTZ: ![[FTZ]] = !{i32 4, !"nvvm-reflect-ftz", i32 1}

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
  %arrayidx = getelementptr inbounds i64, ptr %c, i64 %indvars.iv
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

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.spawn.strategy", i32 3}
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"llvm.loop.unroll.disable"}

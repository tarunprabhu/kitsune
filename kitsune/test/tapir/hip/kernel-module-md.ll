; Check that the tapir target adds the expected module-level metadata to the
; kernel module
;
; RUN: opt %s --tapir=hip --tapir-hip-arch=gfx90a -passes='tapir-lowering<O2>' \
; RUN:     | kit-mbc -S \
; RUN:     | FileCheck %s
;
; The module identifier is generated a specific way. We don't really need it to
; be exactly what it is, but might as well check it.
;
; CHECK: ModuleID = '__kithip_kernel-module-md.ll'
;
; CHECK: target triple = "amdgcn-amd-amdhsa"
;
; CHECK: define {{.*}}@[[F1:__kithip_loop_f1[^(]*]](
; CHECK: define {{.*}}@[[F2:__kithip_loop_f2[^(]*]](
;
; CHECK: !kitsune.device.module.flags = !{![[MDTT:[0-9]+]], ![[MDNAME:[0-9]+]]}
;
; CHECK-DAG: ![[MDTT]] = !{i32 4}
; CHECK-DAG: ![[MDNAME]] = !{!"__kithip_kernel-module-md.ll"}

target triple = "x86_64-pc-linux-gnu"

define void @f1(ptr %c, i32 %n) #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp5 = icmp sgt i32 %n, 0
  br i1 %cmp5, label %preheader, label %forall.sync

preheader:
  %wide.trip.count = zext nneg i32 %n to i64
  br label %forall.detach

forall.detach:
  %indvars.iv = phi i64 [ 0, %preheader ], [ %indvars.iv.next, %forall.inc ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  store i32 %n, ptr %arrayidx, align 4
  reattach within %syncreg, label %forall.inc

forall.inc:
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !0

forall.sync:
  sync within %syncreg, label %forall.end

forall.end:
  ret void
}

define void @f2(ptr %c, i32 %n) #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp5 = icmp sgt i32 %n, 0
  br i1 %cmp5, label %preheader, label %forall.sync

preheader:
  %wide.trip.count = zext nneg i32 %n to i64
  br label %forall.detach

forall.detach:
  %indvars.iv = phi i64 [ 0, %preheader ], [ %indvars.iv.next, %forall.inc ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  store i32 %n, ptr %arrayidx, align 4
  reattach within %syncreg, label %forall.inc

forall.inc:
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !3

forall.sync:
  sync within %syncreg, label %forall.end

forall.end:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}
!3 = !{!3, !1, !2}

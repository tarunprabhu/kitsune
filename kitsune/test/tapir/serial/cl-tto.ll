; Check that the opt command line options are correctly read by the tapir
; targets. Any options that are common to all tapir targets should be tested
; here since the serial tapir target is guaranteed to be built.
;
; RUN: opt --tapir=serial -passes="tapir-lowering<O2>" -o /dev/null %s \
; RUN:      --tapir-verbose 2>&1 \
; RUN:      | FileCheck %s -check-prefixes ALL,COMPILE
;
; RUN: opt --tapir=serial -passes="tapir-lowering<O2>" -o /dev/null %s \
; RUN:      --tapir-verbose --kitrt-verbose 2>&1\
; RUN:      | FileCheck %s -check-prefixes ALL,RUNTIME
;
; RUN: opt --tapir=serial -passes="tapir-lowering<O1>" -o /dev/null %s \
; RUN:      --tapir-verbose 2>&1 \
; RUN:      | FileCheck %s -check-prefixes ALL,O1
;
; RUN: opt --tapir=serial -passes="tapir-lowering<O3>" -o /dev/null %s \
; RUN:      --tapir-verbose 2>&1 \
; RUN:      | FileCheck %s -check-prefixes ALL,O3
;
; RUN: opt --tapir=serial --tapir-verbose -passes='tapir-lowering<O2>' %s \
; RUN:      -o /dev/null --fp-contract=off 2>&1 \
; RUN:      | FileCheck %s -check-prefixes=ALL,STRICT
;
; RUN: opt --tapir=serial --tapir-verbose -passes='tapir-lowering<O2>' %s \
; RUN:      -o /dev/null --fp-contract=on 2>&1 \
; RUN:      | FileCheck %s -check-prefixes=ALL,STANDARD
;
; RUN: opt --tapir=serial --tapir-verbose -passes='tapir-lowering<O2>' %s \
; RUN:      -o /dev/null --fp-contract=fast 2>&1 \
; RUN:      | FileCheck %s -check-prefixes=ALL,FAST
;
; ALL: 'serial' tapir target options
; COMPILE:   Runtime verbose: 1
; RUNTIME:   Runtime verbose: 1
; O1:        Optimization level: O1
; O3:        Optimization level: O3
; STRICT:    FP fusion: strict
; STANDARD:  FP fusion: standard
; FAST:      FP fusion: fast

target triple = "x86_64-unknown-linux-gnu"

define void @f(ptr %c, i32 %n) {
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
!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}

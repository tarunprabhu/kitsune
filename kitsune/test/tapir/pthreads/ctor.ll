; Check that the global ctor calls the appropriate functions in Kitsune's
; runtime depending on the command line arguments passed.
;
; RUN: opt --tapir=pthreads \
; RUN:     -passes='tapir-lowering<O2>,kit-ctors' \
; RUN:     -S %s \
; RUN:     | FileCheck %s -check-prefix DEFAULT
;
; DEFAULT: @llvm.global_ctors = appending global
; DEFAULT-SAME: { i32 65536, ptr @[[CTOR:[.]kitpthr[.]ctor.*]], ptr null }
;
; DEFAULT: define {{.*}} @[[DTOR:[.]kitpthr[.]dtor.*]]{{[ ]*}}(
; DEFAULT: call {{.+}} @llvm.kit.finalize(i32 1024)
;
; DEFAULT: define {{.+}} @[[CTOR]]
; DEFAULT-NEXT: [[ENTRY:.+]]:
; DEFAULT-NEXT: call {{.+}} @llvm.kit.initialize(i32 1024)
; DEFAULT-NEXT: call {{.+}} @llvm.kit.enable.verbose(i8 0)
; DEFAULT-NEXT: call {{.+}}atexit(ptr @[[DTOR]])
; DEFAULT-NEXT: ret void
; DEFAULT-NEXT: }
;
; RUN: opt --tapir=pthreads -passes='tapir-lowering<O2>,kit-ctors' \
; RUN:     -S %s --tapir-verbose \
; RUN:     | FileCheck %s -check-prefix VERBOSE
;
; RUN: opt --tapir=pthreads -passes='tapir-lowering<O2>,kit-ctors' \
; RUN:     -S %s --kitrt-verbose \
; RUN:     | FileCheck %s -check-prefix VERBOSE
;
; VERBOSE-LABEL: define {{.+}} @.kitpthr.ctor
; VERBOSE: call {{.+}} @llvm.kit.enable.verbose(i8 1)
;
; ----------------------------------------------------------------------------

target triple = "x86_64-pc-linux-gnu"

define void @f(ptr %c, i64 %n) {
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
!1 = !{!"tapir.loop.spawn.strategy", i32 4}
!2 = !{!"tapir.loop.target", i32 1024}
!3 = !{!"llvm.loop.unroll.disable"}

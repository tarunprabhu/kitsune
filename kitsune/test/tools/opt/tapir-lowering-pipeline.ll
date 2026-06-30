; Check that the tapir-lowering meta-pass adds the expected passes to the
; pipeline.
;
; ------------------------------------------------------------------------------
; Tapir lowering is available at O0, but only a limited set of passes are run.
; We don't use the text of this file as input here because it will result in a
; failure since loop-spawning will not run.
;
; FIXME: Should we consider not allowing -O0 as an optimization level for
; Kitsune's lowering passes?
;
; RUN: echo "" \
; RUN:     | opt -passes='tapir-lowering<O0>' --tapir=serial \
; RUN:           -debug-pass-manager -disable-output 2>&1 \
; RUN:     | FileCheck %s -check-prefix O0
;
; O0:         Running pass:     TapirToTargetPass
; O0:         Running pass:     AlwaysInlinerPass
; O0:         Running pass:     VerifierPass
;
; ------------------------------------------------------------------------------
; At higher optimization levels, the Kitsune passes that are run are always
; the same. If we ever have optimization passes that are dependent on the
; optimization level, this should be updated
;
; RUN: opt -passes='tapir-lowering<O1>' --tapir=serial -debug-pass-manager %s \
; RUN:     -disable-output 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123S
;
; RUN: opt -passes='tapir-lowering<O2>' --tapir=serial -debug-pass-manager %s \
; RUN:     -disable-output 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123S
;
; RUN: opt -passes='tapir-lowering<O3>' --tapir=serial -debug-pass-manager %s \
; RUN:     -disable-output 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123S
;
; RUN: opt -passes='tapir-lowering<Os>' --tapir=serial -debug-pass-manager %s \
; RUN:     -disable-output 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123S
;
; RUN: not opt -passes='tapir-lowering<Oz>' --tapir=serial -debug-pass-manager \
; RUN:     -disable-output %s 2>&1 \
; RUN:     | FileCheck %s -check-prefix ERROR
;
; ERROR: unsupported optimization level '-Oz'
;
; O123S:      Running pass:     PreLowerPreparePass
; O123S:      Running pass:     SecondaryIVEliminationPass
; O123S:      Running pass:     DeLICMPass
; O123S:      Running pass:     SimplifyCFGPass
; O123S:      Running pass:     LoopSimplifyPass
; O123S:      Running pass:     PreLowerVerificationPass
; O123S:      Running pass:     PreLowerAnnotatePass
; O123S:      Running pass:     SerializePass
; O123S:      Running pass:     LoopSpawningPass
; O123S:      Running pass:     TapirToTargetPass
; O123S:      Running pass:     VerifierPass
;
; ------------------------------------------------------------------------------

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %n = sext i32 %argc to i64
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  call void @ext(i64 %i)
  reattach within %syncreg, label %latch

latch:
  %inc.i = add i64 %i, 1
  %cmp = icmp eq i64 %inc.i, %n
  br i1 %cmp, label %exit, label %header, !llvm.loop !0

exit:
  sync within %syncreg, label %end

end:
  ret i32 0
}

declare void @ext(i64)

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.target", i32 1}
!2 = !{!"tapir.loop.spawn.strategy", i32 1}
!3 = !{!"tapir.loop.lowering.enabled"}

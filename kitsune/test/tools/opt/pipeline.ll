; -----------------------------------------------------------------------------
; If the --tapir option is not provided to opt, neither tapir, nor Kitsune
; passes are run.
;
; RUN: opt -O0 -debug-pass-manager %s -disable-output 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; RUN: opt -O1 -debug-pass-manager %s -disable-output 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; RUN: opt -O2 -debug-pass-manager %s -disable-output 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; RUN: opt -O3 -debug-pass-manager %s -disable-output 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; RUN: opt -Os -debug-pass-manager %s -disable-output 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; RUN: opt -Oz -debug-pass-manager %s -disable-output 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; DEFAULT-NOT: Running pass:     PreLowerPreparePass
; DEFAULT-NOT: Running pass:     SecondaryIVEliminationPass
; DEFAULT-NOT: Running pass:     PrepareTapirLoopsPass
; DEFAULT-NOT: Running pass:     LowerKitReduceIntrinsicsPass
; DEFAULT-NOT: Running pass:     DeLICMPass
; DEFAULT-NOT: Running pass:     PreLowerAnnotatePass
; DEFAULT-NOT: Running pass:     LoopSpawningPass
; DEFAULT-NOT: Running pass:     TapirToTargetPass
; DEFAULT-NOT: Running pass:     PrefetchForDevicePass
; DEFAULT-NOT: Running pass:     EmbLowerKitIntrinsicsEarlyPass
; DEFAULT-NOT: Running pass:     EmbResolveLibDeviceCallsPass
; DEFAULT-NOT: Running pass:     EmbPreparePass
; DEFAULT-NOT: Running pass:     EmbLinkLibDeviceBitcodePass
; DEFAULT-NOT: Running pass:     EmbOptimizePass
; DEFAULT-NOT: Running pass:     RecomputeKernelPropertiesPass
; DEFAULT-NOT: Running pass:     GenerateCtorsPass
;
; -----------------------------------------------------------------------------
; Unlike the frontends, -O0 is allowed with --tapir, even if the tapir target
; is not nolo. In this case, only a limited number of passes are run.
; We don't use the text of this file as input here because it will result in a
; failure since loop-spawning will not run.
;
; FIXME: Should we consider not allowing -O0 as an optimization level if
; --tapir is given. It is not clear what advantage there is to allowing -O0 if
; it is likely to result in a failure.
;
; RUN: echo "" | opt -O0 --tapir=serial -debug-pass-manager -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix O0 %s
;
; O0:      Running pass:     TapirToTargetPass
; O0:      Running pass:     AlwaysInlinerPass
; O0:      Running pass:     VerifierPass
; O0:      Running pass:     BitcodeWriterPass
;
; -----------------------------------------------------------------------------
; If the --tapir option is provided to opt, the Kitsune passes are run at all
; optimization levels.
;
; RUN: opt -O1 --tapir=serial -debug-pass-manager %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix O123SZ %s
;
; RUN: opt -O2 --tapir=serial -debug-pass-manager %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix O123SZ %s
;
; RUN: opt -O3 --tapir=serial -debug-pass-manager %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix O123SZ %s
;
; RUN: opt -Os --tapir=serial -debug-pass-manager %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix O123SZ %s
;
; RUN: opt -Oz --tapir=serial -debug-pass-manager %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix O123SZ %s
;
; The Early* passes run early in the pass pipeline.
; O123S:       Running pass:     EarlyVerificationPass
; O123S:       Running pass:     EarlyAnnotatePass
;
; <KIT-PRE-TAPIR>
; There are no standard pre-tapir passes at this time
; </KIT-PRE-TAPIR>
;
; <KIT-PRE-LOOP-SPAWNING>
; We add LoopSimplify, LoopRotate and LoopLCSSA to the pipeline before
; PrepareReductionLoops, but it is difficult to check for them because they
; match runs of the pass from earlier in the pipeline. PrepareReductionLoops
; will fail if any of these are not run, so something will at least catch it
; if they are ever removed from the pipeline.
; O123SZ:      Running pass:     PreLowerPreparePass
; O123SZ:      Running pass:     SecondaryIVEliminationPass
; O123SZ:      Running pass:     PrepareTapirLoopsPass
; O123SZ:      Running pass:     LowerKitReduceIntrinsicsPass
; O123SZ:      Running pass:     DeLICMPass
; O123SZ:      Running pass:     SimplifyCFGPass
; O123SZ:      Running pass:     LoopSimplifyPass
; O123SZ:      Running pass:     PreLowerVerificationPass
; O123SZ:      Running pass:     PreLowerAnnotatePass
; O123SZ:      Running pass:     SerializePass
; </KIT-PRE-LOOP-SPAWNING>
;
; O123SZ:      Running pass:     LoopSpawningPass
; O123SZ:      Running pass:     TapirToTargetPass
; O123SZ:      Running pass:     GlobalDCEPass
;
; <KIT-POST-TAPIR>
; O123SZ:      Running pass:     PrefetchForDevicePass
; O123SZ:      Running pass:     EmbLowerKitIntrinsicsEarlyPass
; O123SZ:      Running pass:     EmbResolveLibDeviceCallsPass
; O123SZ:      Running pass:     EmbPreparePass
; O123SZ:      Running pass:     EmbLinkLibDeviceBitcodePass
; O123SZ:      Running pass:     EmbOptimizePass
; O123SZ:      Running pass:     RecomputeKernelPropertiesPass
; O123SZ:      Running pass:     GenerateCtorsPass
; </KIT-POST-TAPIR>
;
; O123SZ:      Running pass:     VerifierPass
; O123SZ:      Running pass:     BitcodeWriterPass
;
; -----------------------------------------------------------------------------

define i32 @f(i32 %argc, ptr %argv) {
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

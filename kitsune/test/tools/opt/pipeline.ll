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
; DEFAULT-NOT: Running pass:     TapirLoopAnnotatorPass
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
;
; RUN: opt -O0 --tapir=serial -debug-pass-manager %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix O0 %s
;
; O0:      Running pass:     TapirToTargetPass
; O0-NEXT: Running analysis: TTObjectsAnalysis
; O0-NEXT: Running analysis: LoopAnalysis
; O0-NEXT: Running analysis: DominatorTreeAnalysis
; O0-NEXT: Running analysis: TaskAnalysis
; O0-NEXT: Running pass:     AlwaysInlinerPass
; O0-NEXT: Running pass:     AnnotationRemarksPass
; O0-NEXT: Running analysis: TargetLibraryAnalysis
; O0-NEXT: Running pass:     VerifierPass
; O0-NEXT: Running analysis: VerifierAnalysis
; O0-NEXT: Running pass:     BitcodeWriterPass
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
; The EarlyAnnotatePass runs early in the pass pipeline.
; O123SZ:      Running pass:     EarlyAnnotatePass
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
; O123SZ:      Running pass:     PrepareReductionLoopsPass
; O123SZ:      Running pass:     EarlyCSEPass
; O123SZ:      Running pass:     SimplifyCFGPass
; O123SZ:      Running pass:     InstCombinePass
; O123SZ:      Running pass:     SCCPPass
; O123SZ:      Running pass:     BDCEPass
; O123SZ:      Running pass:     InstCombinePass
; O123SZ:      Running pass:     DSEPass
; O123SZ:      Running pass:     ADCEPass
; O123SZ:      Running pass:     DeLICMPass
; O123SZ:      Running pass:     SimplifyCFGPass
; O123SZ:      Running pass:     LoopSimplifyPass
; O123SZ:      Running pass:     PreLowerVerificationPass
; O123SZ:      Running pass:     PreLowerAnnotate
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

define void @f() {
  ret void
}

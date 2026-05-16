; Check that the Kitsune-lowering meta-pass adds the expected passes to the
; pipeline.
;
; ------------------------------------------------------------------------------
; Kitsune lowering is available at O0, but only a limited set of passes are run.
;
; RUN: opt --tapir=serial -passes='kit-lowering<O0>' -debug-pass-manager %s \
; RUN:     -o /dev/null 2>&1 \
; RUN:     | FileCheck %s -check-prefix O0
;
; O0:      Running pass:     TapirToTargetPass
; O0:      Running pass:     AlwaysInlinerPass
; O0-NEXT: Running analysis: ProfileSummaryAnalysis
; O0-NEXT: Running pass:     VerifierPass
; O0-NEXT: Running analysis: VerifierAnalysis
; O0-NEXT: Running pass:     BitcodeWriterPass
;
; ------------------------------------------------------------------------------
; At higher optimization levels, the Kitsune passes that are run are always
; the same. This may need to change if we ever have optimization-level-dependent
; Kitsune lowering pipelines.
;
; RUN: opt --tapir=serial -passes='kit-lowering<O1>' -debug-pass-manager %s \
; RUN:     -o /dev/null 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123SZ
;
; RUN: opt --tapir=serial -passes='kit-lowering<O2>' -debug-pass-manager %s \
; RUN:     -o /dev/null 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123SZ
;
; RUN: opt --tapir=serial -passes='kit-lowering<O3>' -debug-pass-manager %s \
; RUN:     -o /dev/null 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123SZ
;
; RUN: opt --tapir=serial -passes='kit-lowering<Os>' -debug-pass-manager %s \
; RUN:     -o /dev/null 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123SZ
;
; RUN: opt --tapir=serial -passes='kit-lowering<Oz>' -debug-pass-manager %s \
; RUN:     -o /dev/null 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123SZ
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
; O123SZ-NEXT: Running pass:     LoopSpawningPass
; O123SZ:      Running pass:     TapirToTargetPass
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

define void @f() {
  ret void
}

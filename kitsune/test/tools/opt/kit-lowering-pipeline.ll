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
; O123SZ:      Running pass:     DeLICMPass
; O123SZ-NEXT: Running analysis: ScalarEvolutionAnalysis
; O123SZ-NEXT: Running pass:     SimplifyCFGPass
; O123SZ-NEXT: Running pass:     LoopSimplifyPass
; O123SZ-NEXT: Running pass:     PreLowerVerificationPass
; O123SZ-NEXT: Running analysis: TTObjectsAnalysis
; O123SZ-NEXT: Running analysis: TaskAnalysis
; O123SZ-NEXT: Running analysis: PostDominatorTreeAnalysis
; O123SZ-NEXT: Running pass:     PreLowerAnnotate
; O123SZ-NEXT: Running pass:     SerializePass
; O123SZ-NEXT: Running pass:     LoopSpawningPass
; O123SZ:      Running pass:     TapirToTargetPass
; O123SZ:      Running pass:     PrefetchForDevicePass
; O123SZ-NEXT: Running analysis: TTObjectsAnalysis
; O123SZ-NEXT: Running pass:     EmbLowerKitIntrinsicsLibDevicePass
; O123SZ-NEXT: Running pass:     EmbResolveLibDeviceCallsPass
; O123SZ-NEXT: Running pass:     EmbPreparePass
; O123SZ-NEXT: Running pass:     EmbLinkLibDeviceBitcodePass
; O123SZ-NEXT: Running pass:     EmbOptimizePass
; O123SZ-NEXT: Running pass:     RecomputeKernelPropertiesPass
; O123SZ-NEXT: Running pass:     GenerateCtorsPass
; O123SZ-NEXT: Running pass:     VerifierPass
; O123SZ-NEXT: Running analysis: VerifierAnalysis
; O123SZ-NEXT: Running pass:     BitcodeWriterPass

define void @f() {
  ret void
}

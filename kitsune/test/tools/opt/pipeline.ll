; -----------------------------------------------------------------------------
; If the --tapir option is not provided to opt, neither tapir, nor Kitsune
; passes are run.
;
; RUN: opt -O0 -debug-pass-manager %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; RUN: opt -O1 -debug-pass-manager %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; RUN: opt -O2 -debug-pass-manager %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; RUN: opt -O3 -debug-pass-manager %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; RUN: opt -Os -debug-pass-manager %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; RUN: opt -Oz -debug-pass-manager %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; DEFAULT-NOT: Running pass:     TapirLoopAnnotatorPass
; DEFAULT-NOT: Running pass:     LoopSpawningPass
; DEFAULT-NOT: Running pass:     TapirToTargetPass
; DEFAULT-NOT: Running pass:     PrefetchForDevicePass
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
; O0-NEXT: Running analysis: TapirTargetAnalysis
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
; O123SZ:      Running pass:     PreLowerVerificationPass
; O123SZ-NEXT: Running analysis: TapirTargetAnalysis
; O123SZ-NEXT: Running analysis: TaskAnalysis
; O123SZ-NEXT: Running analysis: PostDominatorTreeAnalysis
; O123SZ-NEXT: Running analysis: ScalarEvolutionAnalysis
; O123SZ-NEXT: Running pass:     AnnotateTapirLoopsPass
; O123SZ-NEXT: Running pass:     SerializePass
; O123SZ-NEXT: Running pass:     LoopSpawningPass
; O123SZ:      Running pass:     TapirToTargetPass
; O123SZ:      Running pass:     GlobalDCEPass
; O123SZ-NEXT: Running pass:     PrefetchForDevicePass
; O123SZ-NEXT: Running pass:     EmbResolveLibDeviceCallsPass
; O123SZ-NEXT: Running pass:     EmbPreparePass
; O123SZ-NEXT: Running pass:     EmbLinkLibDeviceBitcodePass
; O123SZ-NEXT: Running pass:     EmbOptimizePass
; O123SZ-NEXT: Running pass:     RecomputeKernelPropertiesPass
; O123SZ-NEXT: Running pass:     GenerateCtorsPass
; O123SZ-NEXT: Running pass:     VerifierPass
; O123SZ-NEXT: Running analysis: VerifierAnalysis
; O123SZ-NEXT: Running pass:     BitcodeWriterPass
;
; -----------------------------------------------------------------------------

define void @f() {
  ret void
}

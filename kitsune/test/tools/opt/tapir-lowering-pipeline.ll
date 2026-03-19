; Check that the tapir-lowering meta-pass adds the expected passes to the
; pipeline.
;
; ------------------------------------------------------------------------------
; Tapir lowering is available at O0, but only a limited set of passes are run.
;
; RUN: opt -passes='tapir-lowering<O0>' --tapir=serial -debug-pass-manager %s \
; RUN:     -disable-output 2>&1 \
; RUN:     | FileCheck %s -check-prefix O0
;
; O0:      Running pass:     TapirToTargetPass
; O0-NEXT: Running analysis: InnerAnalysisManagerProxy
; O0-NEXT: Running analysis: TTObjectsAnalysis
; O0-NEXT: Running pass:     AlwaysInlinerPass
; O0-NEXT: Running analysis: ProfileSummaryAnalysis
; O0-NEXT: Running pass:     VerifierPass
;
; ------------------------------------------------------------------------------
; At higher optimization levels, the Kitsune passes that are run are always
; the same. If we ever have optimization passes that are dependent on the
; optimization level, this should be updated
;
; RUN: opt -passes='tapir-lowering<O1>' --tapir=serial -debug-pass-manager %s \
; RUN:     -disable-output 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123SZ
;
; RUN: opt -passes='tapir-lowering<O2>' --tapir=serial -debug-pass-manager %s \
; RUN:     -disable-output 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123SZ
;
; RUN: opt -passes='tapir-lowering<O3>' --tapir=serial -debug-pass-manager %s \
; RUN:     -disable-output 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123SZ
;
; RUN: opt -passes='tapir-lowering<Os>' --tapir=serial -debug-pass-manager %s \
; RUN:     -disable-output 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123SZ
;
; RUN: opt -passes='tapir-lowering<Oz>' --tapir=serial -debug-pass-manager %s \
; RUN:     -disable-output 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123SZ
;
; O123SZ:      Running pass:     PreLowerVerificationPass
; O123SZ-NEXT: Running analysis: TTObjectsAnalysis
; O123SZ-NEXT: Running pass:     PreLowerAnnotate
; O123SZ-NEXT: Running pass:     SerializePass
; O123SZ-NEXT: Running pass:     LoopSpawningPass
; O123SZ-NEXT: Running pass:     TapirToTargetPass
; O123SZ-NEXT: Running pass:     IPSCCPPass
; O123SZ-NEXT: Running pass:     CalledValuePropagationPass
; O123SZ-NEXT: Running pass:     GlobalOptPass
; O123SZ-NEXT: Running pass:     DeadArgumentEliminationPass
; O123SZ-NEXT: Running pass:     AlwaysInlinerPass
; O123SZ-NEXT: Running analysis: ProfileSummaryAnalysis
; O123SZ:      Running analysis: GlobalsAA
; O123SZ-NEXT: Running analysis: CallGraphAnalysis
; O123SZ:      Running analysis: LazyCallGraphAnalysis
; O123SZ-NEXT: Running pass:     EliminateAvailableExternallyPass
; O123SZ-NEXT: Running pass:     ReversePostOrderFunctionAttrs
; O123SZ-NEXT: Running pass:     GlobalDCEPass
; O123SZ-NEXT: Running pass:     VerifierPass
;
; ------------------------------------------------------------------------------

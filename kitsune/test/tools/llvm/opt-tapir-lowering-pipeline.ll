; Check that the tapir-lowering meta-pass adds the expected passes to the
; pipeline.
;
; ------------------------------------------------------------------------------
; Tapir lowering is not available at O0.
;
; RUN: not opt -passes='tapir-lowering<O0>' -print-pipeline-passes %s \
; RUN:     -disable-output 2>&1 \
; RUN:     | FileCheck %s -check-prefix O0
;
; O0: tapir-lowering passes require optimization level O1 or higher
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
; O123SZ: Running pass: LoopSpawningPass
; O123SZ-NEXT: Running analysis: TapirTargetAnalysis
; O123SZ-NEXT: Running pass: TapirToTargetPass
; O123SZ-NEXT: Running pass: IPSCCPPass
; O123SZ-NEXT: Running pass: CalledValuePropagationPass
; O123SZ-NEXT: Running pass: GlobalOptPass
; O123SZ-NEXT: Running pass: DeadArgumentEliminationPass
; O123SZ-NEXT: Running pass: AlwaysInlinerPass
; O123SZ-NEXT: Running analysis: ProfileSummaryAnalysis
; O123SZ: Running analysis: GlobalsAA
; O123SZ-NEXT: Running analysis: CallGraphAnalysis
; O123SZ: Running analysis: LazyCallGraphAnalysis
; O123SZ-NEXT: Running pass: EliminateAvailableExternallyPass
; O123SZ-NEXT: Running pass: ReversePostOrderFunctionAttrs
; O123SZ-NEXT: Running pass: GlobalDCEPass
;
; ------------------------------------------------------------------------------

; Check that the Kitsune-lowering meta-pass adds the expected passes to the
; pipeline.
;
; ------------------------------------------------------------------------------
; Kitsune lowering is available at O0, but only a limited set of passes are run.
;
; RUN: opt -passes='kit-lowering<O0>' --tapir=serial -debug-pass-manager %s \
; RUN:     -o /dev/null 2>&1 \
; RUN:     | FileCheck %s -check-prefix O0
;
; O0:      Running pass:     LowerMobileIntrinsicsPass
; O0-NEXT: Running analysis: TapirTargetAnalysis
; O0:      Running pass:     StripKitsuneAddrSpacePass
; O0-NEXT: Running pass:     TapirToTargetPass
; O0-NEXT: Running pass:     AlwaysInlinerPass
; O0-NEXT: Running analysis: ProfileSummaryAnalysis
; O0-NEXT: Running pass:     LowerKitsuneRuntimeIntrinsicsPass
; O0-NEXT: Running pass:     VerifierPass
; O0-NEXT: Running analysis: VerifierAnalysis
; O0-NEXT: Running pass:     BitcodeWriterPass
;
; ------------------------------------------------------------------------------
; At higher optimization levels, the Kitsune passes that are run are always
; the same. If we ever have optimization passes that are dependent on the
; optimization level, this should be updated
;
; RUN: opt -passes='kit-lowering<O1>' --tapir=serial -debug-pass-manager %s \
; RUN:     -o /dev/null 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123SZ
;
; RUN: opt -passes='kit-lowering<O2>' --tapir=serial -debug-pass-manager %s \
; RUN:     -o /dev/null 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123SZ
;
; RUN: opt -passes='kit-lowering<O3>' --tapir=serial -debug-pass-manager %s \
; RUN:     -o /dev/null 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123SZ
;
; RUN: opt -passes='kit-lowering<Os>' --tapir=serial -debug-pass-manager %s \
; RUN:     -o /dev/null 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123SZ
;
; RUN: opt -passes='kit-lowering<Oz>' --tapir=serial -debug-pass-manager %s \
; RUN:     -o /dev/null 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123SZ
;
; O123SZ:      Running pass:     LowerMobileIntrinsicsPass
; O123SZ-NEXT: Running analysis: TapirTargetAnalysis
; O123SZ:      Running pass:     StripKitsuneAddrSpacePass
; O123SZ:      Running pass:     LoopSpawningPass
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
; O123SZ-NEXT: Running pass:     EmbResolveLibDeviceCallsPass
; O123SZ-NEXT: Running pass:     EmbPreparePass
; O123SZ-NEXT: Running pass:     EmbLinkLibDeviceBitcodePass
; O123SZ-NEXT: Running pass:     EmbOptimizePass
; O123SZ-NEXT: Running pass:     RecomputeKernelPropertiesPass
; O123SZ-NEXT: Running pass:     GenerateCtorsPass
; O123SZ-NEXT: Running pass:     LowerKitsuneRuntimeIntrinsicsPass
; O123SZ-NEXT: Running pass:     VerifierPass
; O123SZ-NEXT: Running analysis: VerifierAnalysis
; O123SZ-NEXT: Running pass:     BitcodeWriterPass

// If the --tapir=nolo option is provided without optimizations, neither tapir,
// nor Kitsune, passes are run.
//
// RUN: %kitxx --tapir=nolo -O0 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O0
//
// O0:      Running pass:     AlwaysInlinerPass
// O0-NEXT: Running analysis: ProfileSummaryAnalysis
// O0-NEXT: Running pass:     CoroConditionalWrapper
// O0-NEXT: Running pass:     VerifierPass
// O0-NEXT: Running pass:     BitcodeWriterPass
//
// -----------------------------------------------------------------------------
// If the --tapir argument is provided, all Tapir and Kitsune passes should run.
//
// RUN: %kitxx --tapir=serial -O1 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O123SZ
//
// RUN: %kitxx --tapir=serial -O2 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O123SZ
//
// RUN: %kitxx --tapir=serial -O3 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O123SZ
//
// RUN: %kitxx --tapir=serial -Os -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O123SZ
//
// RUN: %kitxx --tapir=serial -Oz -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O123SZ
//
// O123SZ:      Running pass:     CreateEmbBitcodePass
// O123SZ-NEXT: Running analysis: TapirTargetAnalysis
// O123SZ-NEXT: Running pass:     LoopSpawningPass
// O123SZ-NEXT: Running pass:     TapirToTargetPass
// O123SZ-NEXT: Running pass:     IPSCCPPass
// O123SZ-NEXT: Running pass:     CalledValuePropagationPass
// O123SZ-NEXT: Running pass:     GlobalOptPass
// O123SZ-NEXT: Running pass:     DeadArgumentEliminationPass
// O123SZ-NEXT: Running pass:     AlwaysInlinerPass
// O123SZ-NEXT: Running pass:     RequireAnalysisPass
// O123SZ-NEXT: Running pass:     EliminateAvailableExternallyPass
// O123SZ-NEXT: Running pass:     ReversePostOrderFunctionAttrs
// O123SZ-NEXT: Running pass:     GlobalDCEPass
// O123SZ-NEXT: Running pass:     PrefetchingPass
// O123SZ-NEXT: Running pass:     EmbResolveLibDeviceCallsPass
// O123SZ-NEXT: Running pass:     EmbPreparePass
// O123SZ-NEXT: Running pass:     EmbLinkLibDeviceBitcodePass
// O123SZ-NEXT: Running pass:     EmbOptimizePass
// O123SZ-NEXT: Running pass:     RecomputeKernelPropertiesPass
// O123SZ-NEXT: Running pass:     GenerateCtorsPass
// O123SZ-NEXT: Running pass:     VerifierPass
// O123SZ-NEXT: Running pass:     BitcodeWriterPass

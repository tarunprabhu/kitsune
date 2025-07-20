// -----------------------------------------------------------------------------
// If the tapir target is nolo, the behavior is similar to the regular pipeline
// i.e. loop spawning is not run and neither are any Kitsune passes.
//
// RUN: %kitcc -O2 --tapir=nolo -Xlinker --lto-emit-llvm -o /dev/null %s \
// RUN:     -flto -Xlinker --lto-debug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix NOLO
//
// NOLO:      Running pass:     VerifierPass
// NOLO-NOT:  Running pass:     LoopSpawning
// NOLO-NOT:  Running pass:     LowerRuntimeIntrinsicsPass
// NOLO:      Running pass:     VerifierPass
// NOLO-NEXT: Running analysis: VerifierAnalysis
//
// -----------------------------------------------------------------------------
// All Kitsune, and Tapir, passes should run during the postlink phase of LTO.
//
// RUN: %kitcc -O2 --tapir=serial -Xlinker --lto-emit-llvm -o /dev/null %s \
// RUN:     -flto -Xlinker --lto-debug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O23SZ
//
// RUN: %kitcc -O3 --tapir=serial -Xlinker --lto-emit-llvm -o /dev/null %s \
// RUN:     -flto -Xlinker --lto-debug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O23SZ
//
// RUN: %kitcc -Os --tapir=serial -Xlinker --lto-emit-llvm -o /dev/null %s \
// RUN:     -flto -Xlinker --lto-debug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O23SZ
//
// RUN: %kitcc -Oz --tapir=serial -Xlinker --lto-emit-llvm -o /dev/null %s \
// RUN:     -flto -Xlinker --lto-debug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O23SZ
//
// -----------------------------------------------------------------------------
//
// O23SZ:      Running pass:     CreateEmbBitcodePass
// O23SZ-NEXT: Running analysis: TapirTargetAnalysis
// O23SZ-NEXT: Running pass:     LoopSpawningPass
// O23SZ-NEXT: Running pass:     TapirToTargetPass
// O23SZ-NEXT: Running pass:     IPSCCPPass
// O23SZ-NEXT: Running pass:     CalledValuePropagationPass
// O23SZ-NEXT: Running pass:     GlobalOptPass
// O23SZ-NEXT: Running pass:     DeadArgumentEliminationPass
// O23SZ-NEXT: Running pass:     AlwaysInlinerPass
// O23SZ-NEXT: Running pass:     RequireAnalysisPass
// O23SZ-NEXT: Running pass:     EliminateAvailableExternallyPass
// O23SZ-NEXT: Running pass:     ReversePostOrderFunctionAttrs
// O23SZ-NEXT: Running pass:     GlobalDCEPass
// O23SZ-NEXT: Running pass:     PrefetchingPass
// O23SZ-NEXT: Running pass:     EmbResolveLibDeviceCallsPass
// O23SZ-NEXT: Running pass:     EmbPreparePass
// O23SZ-NEXT: Running pass:     EmbLinkLibDeviceBitcodePass
// O23SZ-NEXT: Running pass:     EmbOptimizePass
// O23SZ-NEXT: Running pass:     RecomputeKernelPropertiesPass
// O23SZ-NEXT: Running pass:     GenerateCtorsPass
// O23SZ-NEXT: Running pass:     VerifierPass
// O23SZ-NEXT: Running analysis: VerifierAnalysis

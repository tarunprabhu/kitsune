// If the --tapir=none option is provided without optimizations, the mandatory
// Kitsune passes should be run. However, none of the tapir passes are run.
//
// RUN: %kitxx --tapir=none -O0 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O0
//
// O0:      Running pass:     LowerMobileIntrinsicsPass
// O0-NEXT: Running analysis: TapirTargetAnalysis
// O0-NEXT: Running pass:     StripKitsuneAddrSpacePass
// O0-NEXT: Running pass:     LowerKitsuneRuntimeIntrinsicsPass
// O0-NEXT: Running pass:     VerifierPass
//
// -----------------------------------------------------------------------------
// If the --tapir argument is provided, all Tapir and Kitsune passes should run.
//
// RUN: %kitxx --tapir=serial -O1 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O23SZ
//
// RUN: %kitxx --tapir=serial -O2 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O23SZ
//
// RUN: %kitxx --tapir=serial -O3 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O23SZ
//
// RUN: %kitxx --tapir=serial -Os -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O23SZ
//
// RUN: %kitxx --tapir=serial -Oz -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O23SZ
//
// O23SZ:      Running pass:     LowerMobileIntrinsicsPass
// O23SZ-NEXT: Running analysis: TapirTargetAnalysis
// O23SZ-NEXT: Running pass:     StripKitsuneAddrSpacePass
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
// O23SZ-NEXT: Running pass:     EmbResolveLibDeviceCallsPass
// O23SZ-NEXT: Running pass:     EmbPreparePass
// O23SZ-NEXT: Running pass:     EmbLinkLibDeviceBitcodePass
// O23SZ-NEXT: Running pass:     EmbOptimizePass
// O23SZ-NEXT: Running pass:     RecomputeKernelPropertiesPass
// O23SZ-NEXT: Running pass:     GenerateCtorsPass
// O23SZ-NEXT: Running pass:     LowerKitsuneRuntimeIntrinsicsPass
// O23SZ-NEXT: Running analysis: TargetLibraryAnalysis
// O23SZ-NEXT: Running pass:     VerifierPass

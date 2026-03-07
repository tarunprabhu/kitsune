// -----------------------------------------------------------------------------
// If the tapir target is nolo, the behavior is similar to the regular pipeline
// i.e. loop spawning is not run and neither are any Kitsune passes.
//
// RUN: %kitcc -flto -O2 --tapir=nolo -o /dev/null %s %sysroot \
// RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
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
// RUN: %kitcc -flto -O2 --tapir=serial -o /dev/null %s %sysroot \
// RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
// RUN:     | FileCheck %s -check-prefix O23SZ
//
// RUN: %kitcc -flto -O3 --tapir=serial -o /dev/null %s %sysroot \
// RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
// RUN:     | FileCheck %s -check-prefix O23SZ
//
// RUN: %kitcc -flto -Os --tapir=serial -o /dev/null %s %sysroot \
// RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
// RUN:     | FileCheck %s -check-prefix O23SZ
//
// RUN: %kitcc -flto -Oz --tapir=serial -o /dev/null %s %sysroot \
// RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
// RUN:     | FileCheck %s -check-prefix O23SZ
//
// -----------------------------------------------------------------------------
//
// O23SZ:      Running pass:     PreLowerVerificationPass
// O23SZ-NEXT: Running analysis: TapirTargetAnalysis
// O23SZ-NEXT: Running pass:     AnnotateTapirLoopsPass
// O23SZ-NEXT: Running pass:     SerializePass
// O23SZ-NEXT: Running pass:     LoopSpawningPass
// O23SZ-NEXT: Running pass:     TapirToTargetPass
// O23SZ:      Running pass:     PrefetchForDevicePass
// O23SZ-NEXT: Running pass:     EmbResolveLibDeviceCallsPass
// O23SZ-NEXT: Running pass:     EmbPreparePass
// O23SZ-NEXT: Running pass:     EmbLinkLibDeviceBitcodePass
// O23SZ-NEXT: Running pass:     EmbOptimizePass
// O23SZ-NEXT: Running pass:     RecomputeKernelPropertiesPass
// O23SZ-NEXT: Running pass:     GenerateCtorsPass
// O23SZ-NEXT: Running pass:     VerifierPass
// O23SZ-NEXT: Running analysis: VerifierAnalysis

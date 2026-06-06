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
// The Kitsune (and Tapir) lowering passes should run during the postlink phase
// of LTO. But the non-lowering passes should not run.
//
// RUN: %kitcc -flto -O2 --tapir=serial -o /dev/null %s %sysroot \
// RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
// RUN:     | FileCheck %s -check-prefix O23S
//
// RUN: %kitcc -flto -O3 --tapir=serial -o /dev/null %s %sysroot \
// RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
// RUN:     | FileCheck %s -check-prefix O23S
//
// RUN: %kitcc -flto -Os --tapir=serial -o /dev/null %s %sysroot \
// RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
// RUN:     | FileCheck %s -check-prefix O23S
//
// RUN: not %kitcc -flto -Oz --tapir=serial -o /dev/null %s %sysroot \
// RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
// RUN:     | FileCheck %s -check-prefix ERROR
//
// -----------------------------------------------------------------------------
//
// O23S-NOT:  Running pass:     EarlyAnnotatePass
//
// O23S:      Running pass:     PreLowerVerificationPass
// O23S-NEXT: Running analysis: TTObjectsAnalysis
// O23S-NEXT: Running pass:     PreLowerAnnotate
// O23S-NEXT: Running pass:     SerializePass
// O23S-NEXT: Running pass:     LoopSpawningPass
// O23S-NEXT: Running pass:     TapirToTargetPass
// O23S:      Running pass:     PrefetchForDevicePass
// O23S-NEXT: Running pass:     EmbLowerKitIntrinsicsEarlyPass
// O23S-NEXT: Running pass:     EmbResolveLibDeviceCallsPass
// O23S-NEXT: Running pass:     EmbPreparePass
// O23S-NEXT: Running pass:     EmbLinkLibDeviceBitcodePass
// O23S-NEXT: Running pass:     EmbOptimizePass
// O23S-NEXT: Running pass:     RecomputeKernelPropertiesPass
// O23S-NEXT: Running pass:     GenerateCtorsPass
// O23S-NEXT: Running pass:     VerifierPass
// O23S-NEXT: Running analysis: VerifierAnalysis
//
// ERROR: unsupported optimization level '-Oz'

void f() {}

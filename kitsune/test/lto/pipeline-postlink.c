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
// O23S-NOT:  Running pass:     EarlyVerificationPass
// O23S-NOT:  Running pass:     EarlyAnnotatePass
// O23S-NOT:  Running pass:     PrepareTapirLoopsPass
// O23S-NOT:  Running pass:     LowerReduceIntrinsicsPass
//
// O23S:      Running pass:     NormalizeLoopControlBlocksPass
// O23S:      Running pass:     SecondaryIVEliminationPass
// O23S:      Running pass:     DeLICMPass
// O23S:      Running pass:     SimplifyCFGPass
// O23S:      Running pass:     LoopSimplifyPass
// O23S:      Running pass:     PreLowerVerificationPass
// O23S:      Running pass:     PreLowerAnnotatePass
// O23S:      Running pass:     SerializePass
// O23S:      Running pass:     LoopSpawningPass
// O23S:      Running pass:     HoistAllocasPass
// O23S:      Running pass:     EmbHoistAllocasPass
// O23S:      Running pass:     TapirToTargetPass
// O23S:      Running pass:     PrefetchForDevicePass
// O23S:      Running pass:     EmbLowerWarpIntrinsicsPass
// O23S:      Running pass:     EmbLowerIntrinsicsPass
// O23S:      Running pass:     EmbResolveLibDeviceCallsPass
// O23S:      Running pass:     EmbPreparePass
// O23S:      Running pass:     EmbLinkLibDeviceBitcodePass
// O23S:      Running pass:     EmbOptimizePass
// O23S:      Running pass:     RecomputeKernelPropertiesPass
// O23S:      Running pass:     GenerateCtorsPass
// O23S:      Running pass:     VerifierPass
// O23S:      Running analysis: VerifierAnalysis
//
// ERROR: unsupported optimization level '-Oz'

#include <kitsune.h>

void ext(long);

int main(int argc, char *argv[]) {
  forall (long i = 0; i < argc; ++i)
    ext(i);
  return 0;
}

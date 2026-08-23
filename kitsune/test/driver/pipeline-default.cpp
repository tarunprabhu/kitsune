// If the --tapir=nolo option is provided without optimizations, neither tapir,
// nor Kitsune, passes are run.
//
// RUN: %kitxx --tapir=nolo -O0 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O0
//
// O0-NOT:     Running pass:     SecondaryIVEliminationPass
// O0-NOT:     Running pass:     PrepareTapirLoopsPass
// O0-NOT:     Running pass:     LowerKitReduceIntrinsicsPass
// O0-NOT:     Running pass:     DeLICMPass
// O0-NOT:     Running pass:     NormalizeLoopControlBlocksPass
// O0-NOT:     Running pass:     PreLowerAnnotatePass
// O0-NOT:     Running pass:     LoopSpawningPass
// O0-NOT:     Running pass:     TapirToTargetPass
// O0-NOT:     Running pass:     PrefetchForDevicePass
// O0-NOT:     Running pass:     EmbLowerKitIntrinsicsEarlyPass
// O0-NOT:     Running pass:     EmbResolveLibDeviceCallsPass
// O0-NOT:     Running pass:     EmbPreparePass
// O0-NOT:     Running pass:     EmbLinkLibDeviceBitcodePass
// O0-NOT:     Running pass:     EmbOptimizePass
// O0-NOT:     Running pass:     RecomputeKernelPropertiesPass
// O0-NOT:     Running pass:     GenerateCtorsPass
//
// -----------------------------------------------------------------------------
// If the --tapir argument is provided, all Tapir and Kitsune passes should run.
//
// RUN: %kitxx --tapir=serial -O1 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O123S
//
// RUN: %kitxx --tapir=serial -O2 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O123S
//
// RUN: %kitxx --tapir=serial -O3 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O123S
//
// RUN: %kitxx --tapir=serial -Os -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O123S
//
// RUN: not %kitxx --tapir=serial -Oz -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix ERROR
//
// The Early* passes run early in the pass pipeline.
// O123S:      Running pass:     EarlyVerificationPass
// O123S:      Running pass:     EarlyAnnotatePass
//
// <KIT-PRE-TAPIR>
// There are no standard pre-tapir passes at this time
// </KIT-PRE-TAPIR>
//
// <KIT-PRE-LOOP-SPAWNING>
// O123S:      Running pass:     NormalizeLoopControlBlocksPass
// O123S:      Running pass:     SecondaryIVEliminationPass
// O123S:      Running pass:     PrepareTapirLoopsPass
// O123S-NOT:  Running pass:     InstrumentPass
// O123S:      Running pass:     LowerKitWarpIntrinsicsPass
// O123S:      Running pass:     LowerKitReduceIntrinsicsPass
// O123S:      Running pass:     DeLICMPass
// O123S:      Running pass:     SimplifyCFGPass
// O123S:      Running pass:     LoopSimplifyPass
// O123S:      Running pass:     PreLowerVerificationPass
// O123S:      Running pass:     PreLowerAnnotatePass
// O123S:      Running pass:     SerializePass
// </KIT-PRE-LOOP-SPAWNING>
//
// O123S:      Running pass:     LoopSpawningPass
//
// <KIT-POST-LOOP-SPAWNING>
// O123S:      Running pass:     HoistAllocasPass
// O123S:      Running pass:     EmbHoistAllocasPass
// </KIT-POST-LOOP-SPAWNING>
//
// O123S:      Running pass:     TapirToTargetPass
// O123S:      Running pass:     GlobalDCEPass
//
// <KIT-POST-TAPIR>
// O123S:      Running pass:     PrefetchForDevicePass
// O123S:      Running pass:     EmbLowerKitIntrinsicsEarlyPass
// O123S:      Running pass:     EmbResolveLibDeviceCallsPass
// O123S:      Running pass:     EmbPreparePass
// O123S:      Running pass:     EmbLinkLibDeviceBitcodePass
// O123S:      Running pass:     EmbOptimizePass
// O123S:      Running pass:     RecomputeKernelPropertiesPass
// O123S:      Running pass:     GenerateCtorsPass
// </KIT-POST-TAPIR>
//
// O123S:      Running pass:     VerifierPass
// O123S:      Running pass:     BitcodeWriterPass
//
// ERROR: unsupported optimization level '-Oz'
//
// -----------------------------------------------------------------------------
// The instrumentation pass will only run if instrumentation is explicitly
// enabled.
//
// RUN: %kitxx --tapir=serial -O1 -c -emit-llvm -o /dev/null %s \
// RUN:     --kit-instr=timer \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix INSTR
//
// RUN: %kitxx --tapir=serial -O2 -c -emit-llvm -o /dev/null %s \
// RUN:     --kit-instr=generic \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix INSTR
//
// RUN: %kitxx --tapir=serial -O3 -c -emit-llvm -o /dev/null %s \
// RUN:     --kit-instr=generic \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix INSTR
//
// RUN: %kitxx --tapir=serial -Os -c -emit-llvm -o /dev/null %s \
// RUN:     --kit-instr=generic,timer \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix INSTR
//
// INSTR:      Running pass:     PrepareTapirLoopsPass
// INSTR:      Running pass:     InstrumentPass
//
// -----------------------------------------------------------------------------

#include <kitsune.h>

extern "C" void ext(long);

void f(long n) {
  forall (long i = 0; i < n; ++i)
    ext(i);
}

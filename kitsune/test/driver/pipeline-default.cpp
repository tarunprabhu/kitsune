// If the --tapir=nolo option is provided without optimizations, neither tapir,
// nor Kitsune, passes are run.
//
// RUN: %kitxx --tapir=nolo -O0 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O0
//
// O0-NOT:     Running pass:       SerializePass
// O0-NOT:     Running pass:       PrepareTapirLoopsPass
// O0-NOT:     Running pass:       LocalizeReductionsPass
// O0-NOT:     Running pass:       InstrumentPass
// O0-NOT:     Running pass:       SecondaryIVEliminationPass
// O0-NOT:     Running pass:       PreLowerVerificationPass
// O0-NOT:     Running pass:       PreLowerAnnotatePass
// O0-NOT:     Running pass:       TapirLoopNormalizePass
// O0-NOT:     Running pass:       LoopSpawningPass
// O0-NOT:     Running pass:       TapirToTargetPass
// O0-NOT:     Running pass:       HoistAllocasPass
// O0-NOT:     Running pass:       EmbHoistAllocasPass
// O0-NOT:     Running pass:       PrefetchForDevicePass
// O0-NOT:     Running pass:       EmbLowerIntrinsicsPass
// O0-NOT:     Running pass:       EmbResolveLibDeviceCallsPass
// O0-NOT:     Running pass:       EmbPreparePass
// O0-NOT:     Running pass:       EmbLinkLibDeviceBitcodePass
// O0-NOT:     Running pass:       EmbOptimizePass
// O0-NOT:     Running pass:       RecomputeKernelPropertiesPass
// O0-NOT:     Running pass:       GenerateCtorsPass
//
// -----------------------------------------------------------------------------
// If the --tapir argument is provided, all Tapir and Kitsune passes should run.
//
// RUN: %kitxx --tapir=serial -O1 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O123
//
// RUN: %kitxx --tapir=serial -O2 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O123
//
// RUN: %kitxx --tapir=serial -O3 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O123
//
// RUN: not %kitxx --tapir=serial -Os -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix ERROR
//
// RUN: not %kitxx --tapir=serial -Oz -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix ERROR
//
// ... pipeline early
// O123:       Running pass:       EarlyVerificationPass
// O123:       Running pass:       EarlyAnnotatePass
//
// ... pipeline prepare
// O123:       Running pass:       SerializePass
// O123:       Running pass:       PrepareTapirLoopsPass
// O123:       Running pass:       LocalizeReductionsPass
// O123-NOT:   Running pass:       InstrumentPass
//
// ... pipeline pre-vectorize
// O123:       Running pass:       SecondaryIVEliminationPass
//
// ... pipeline pre-tapir
// There are no standard pre-tapir passes at this time
//
// ... pipeline pre-loop-spawning
// O123:       Running pass:       SecondaryIVEliminationPass
// O123:       Running pass:       PreLowerVerificationPass
// O123:       Running pass:       PreLowerAnnotatePass
// O123:       Running pass:       TapirLoopNormalizePass
//
// O123:       Running pass:       LoopSpawningPass
//
// ... pipeline post-loop-spawning
// O123:       Running pass:       HoistAllocasPass
// O123:       Running pass:       EmbHoistAllocasPass
//
// O123:       Running pass:       TapirToTargetPass
//
// ... pipeline post-tapir
// O123:       Running pass:       PrefetchForDevicePass
// O123:       Running pass:       EmbLowerIntrinsicsPass
// O123:       Running pass:       EmbResolveLibDeviceCallsPass
// O123:       Running pass:       EmbPreparePass
// O123:       Running pass:       EmbLinkLibDeviceBitcodePass
// O123:       Running pass:       EmbOptimizePass
// O123:       Running pass:       RecomputeKernelPropertiesPass
// O123:       Running pass:       GenerateCtorsPass
//
// O123:       Running pass:       VerifierPass
// O123:       Running pass:       BitcodeWriterPass
//
// ERROR: unsupported optimization level '-O{{[sz]}}'
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
// INSTR:      Running pass:       PrepareTapirLoopsPass
// INSTR:      Running pass:       InstrumentPass
//
// -----------------------------------------------------------------------------

#include <kitsune.h>

extern "C" void ext(long);

void f(long n) {
  forall (long i = 0; i < n; ++i)
    ext(i);
}

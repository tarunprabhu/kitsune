// The Kitsune (Tapir) lowering passes should not be run during the prelink
// phase of LTO, but the non-lowering passes should be run.
//
// -----------------------------------------------------------------------------
//
// RUN: %kitcc -O1 --tapir=serial -c -emit-llvm -o /dev/null %s \
// RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s --check-prefix=O123
//
// RUN: %kitcc -O2 --tapir=serial -c -emit-llvm -o /dev/null %s \
// RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s --check-prefix=O123
//
// RUN: %kitcc -O3 --tapir=serial -c -emit-llvm -o /dev/null %s \
// RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s --check-prefix=O123
//
// RUN: not %kitcc -Os --tapir=serial -c -emit-llvm -o /dev/null %s \
// RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s --check-prefix=ERROR
//
// RUN: not %kitcc -Oz --tapir=serial -c -emit-llvm -o /dev/null %s \
// RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s --check-prefix=ERROR
//
// -----------------------------------------------------------------------------
//
// ... pipeline early
// O123:       Running pass:       EarlyVerificationPass
// O123:       Running pass:       EarlyAnnotatePass
//
// ... pipeline prepare
// O123:       Running pass:       SerializePass
// O123:       Running pass:       GuardLoopPass
// O123:       Running pass:       PrepareTapirLoopsPass
// O123:       Running pass:       LocalizeReductionsPass
// O123-NOT:   Running pass:       InstrumentPass
//
// ... pipeline pre-vectorize
// O123:       Running pass:       SecondaryIVEliminationPass
//
// O123-NOT:   Running pass:       SecondaryIVEliminationPass
// O123-NOT:   Running pass:       PreLowerVerificationPass
// O123-NOT:   Running pass:       PreLowerAnnotatePass
// O123-NOT:   Running pass:       TapirLoopNormalizePass
// O123-NOT:   Running pass:       LoopSpawningPass
// O123-NOT:   Running pass:       HoistAllocasPass
// O123-NOT:   Running pass:       EmbHoistAllocasPass
// O123-NOT:   Running pass:       EmbLowerIntrinsicsPass
// O123-NOT:   Running pass:       EmbResolveLibDeviceCallsPass
// O123-NOT:   Running pass:       EmbPreparePass
// O123-NOT:   Running pass:       EmbLinkLibDeviceBitcodePass
// O123-NOT:   Running pass:       EmbOptimizePass
// O123-NOT:   Running pass:       RecomputeKernelPropertiesPass
// O123-NOT:   Running pass:       GenerateCtorsPass
// O123-NOT:   Running pass:       LowerRuntimeIntrinsicsPass
//
// ERROR: unsupported optimization level '-O{{[sz]}}'
//
// -----------------------------------------------------------------------------
// The instrumentation pass will only run if instrumentation is explicitly
// enabled.
//
// RUN: %kitcc -O2 --tapir=serial -c -emit-llvm -o /dev/null %s \
// RUN:     --kit-instr=generic \
// RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s --check-prefix=INSTR
//
// RUN: %kitcc -O3 --tapir=serial -c -emit-llvm -o /dev/null %s \
// RUN:     --kit-instr=timer \
// RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s --check-prefix=INSTR
//
// INSTR:      Running pass:       PrepareTapirLoopsPass
// INSTR:      Running pass:       LocalizeReductionsPass
// INSTR:      Running pass:       InstrumentPass
//
// -----------------------------------------------------------------------------

#include <kitsune.h>

void ext(long);

int main(int argc, char *argv[]) {
  forall (long i = 0; i < argc; ++i)
    ext(i);
  return 0;
}

// The Kitsune (Tapir) lowering passes should not be run during the prelink
// phase of LTO, but the non-lowering passes should be run.
//
// -----------------------------------------------------------------------------
//
// RUN: %kitcc -O2 --tapir=serial -c -emit-llvm -o /dev/null %s \
// RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s --check-prefix=O23
//
// RUN: %kitcc -O3 --tapir=serial -c -emit-llvm -o /dev/null %s \
// RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s --check-prefix=O23
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
// O23:        Running pass:      EarlyVerificationPass
// O23:        Running pass:      EarlyAnnotatePass
// O23:        Running pass:      SerializePass
// O23:        Running pass:      PrepareTapirLoopsPass
// O23:        Running pass:      LocalizeReductionsPass
// O23:        Running pass:      SecondaryIVEliminationPass
// O23-NOT:    Running pass:      InstrumentPass
//
// O23-NOT:    Running pass:      SecondaryIVEliminationPass
// O23-NOT:    Running pass:      PreLowerVerificationPass
// O23-NOT:    Running pass:      PreLowerAnnotatePass
// O23-NOT:    Running pass:      TapirLoopNormalizePass
// O23-NOT:    Running pass:      LoopSpawningPass
// O23-NOT:    Running pass:      HoistAllocasPass
// O23-NOT:    Running pass:      EmbHoistAllocasPass
// O23-NOT:    Running pass:      EmbLowerIntrinsicsPass
// O23-NOT:    Running pass:      EmbResolveLibDeviceCallsPass
// O23-NOT:    Running pass:      EmbPreparePass
// O23-NOT:    Running pass:      EmbLinkLibDeviceBitcodePass
// O23-NOT:    Running pass:      EmbOptimizePass
// O23-NOT:    Running pass:      RecomputeKernelPropertiesPass
// O23-NOT:    Running pass:      GenerateCtorsPass
// O23-NOT:    Running pass:      LowerRuntimeIntrinsicsPass
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
// INSTR:      Running pass:      PrepareTapirLoopsPass
// INSTR:      Running pass:      InstrumentPass
//
// -----------------------------------------------------------------------------

#include <kitsune.h>

void ext(long);

int main(int argc, char *argv[]) {
  forall (long i = 0; i < argc; ++i)
    ext(i);
  return 0;
}

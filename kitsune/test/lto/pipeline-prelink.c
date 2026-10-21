// The Kitsune (Tapir) lowering passes should not be run during the prelink
// phase of LTO, but the non-lowering passes should be run.
//
// -----------------------------------------------------------------------------
//
// RUN: %kitcc -O2 --tapir=serial -c -emit-llvm -o /dev/null %s \
// RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s
//
// RUN: %kitcc -O3 --tapir=serial -c -emit-llvm -o /dev/null %s \
// RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s
//
// RUN: %kitcc -Os --tapir=serial -c -emit-llvm -o /dev/null %s \
// RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s
//
// RUN: not %kitcc -Oz --tapir=serial -c -emit-llvm -o /dev/null %s \
// RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s --check-prefix=ERROR
//
// -----------------------------------------------------------------------------
//
// CHECK:      Running pass:      EarlyVerificationPass
// CHECK:      Running pass:      EarlyAnnotatePass
// CHECK:      Running pass:      SerializePass
// CHECK:      Running pass:      PrepareTapirLoopsPass
// CHECK:      Running pass:      ComputeGPULoopBlockSizePass
// CHECK:      Running pass:      SecondaryIVEliminationPass
// CHECK:      Running pass:      SecondaryIVEliminationPass
// CHECK-NOT:  Running pass:      InstrumentPass
//
// CHECK-NOT:  Running pass:      SecondaryIVEliminationPass
// CHECK-NOT:  Running pass:      LowerIntrinsicsEarlyPass
// CHECK-NOT:  Running pass:      PreLowerVerificationPass
// CHECK-NOT:  Running pass:      PreLowerAnnotatePass
// CHECK-NOT:  Running pass:      TapirLoopNormalizePass
// CHECK-NOT:  Running pass:      LoopSpawningPass
// CHECK-NOT:  Running pass:      HoistAllocasPass
// CHECK-NOT:  Running pass:      EmbHoistAllocasPass
// CHECK-NOT:  Running pass:      EmbFinalizeReductionsPass
// CHECK-NOT:  Running pass:      EmbLowerIntrinsicsPass
// CHECK-NOT:  Running pass:      EmbResolveLibDeviceCallsPass
// CHECK-NOT:  Running pass:      EmbPreparePass
// CHECK-NOT:  Running pass:      EmbLinkLibDeviceBitcodePass
// CHECK-NOT:  Running pass:      EmbOptimizePass
// CHECK-NOT:  Running pass:      RecomputeKernelPropertiesPass
// CHECK-NOT:  Running pass:      GenerateCtorsPass
// CHECK-NOT:  Running pass:      LowerRuntimeIntrinsicsPass
//
// ERROR: unsupported optimization level '-Oz'
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
// RUN: %kitcc -Os --tapir=serial -c -emit-llvm -o /dev/null %s \
// RUN:     --kit-instr=timer,generic \
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

// None of the Kitsune, or Tapir, passes should run during the prelink phase,
// regardless of the specified optimization level.
//
// -----------------------------------------------------------------------------
// Only the nolo tapir target is allowed at -O0.
//
// RUN: %kitcc -O2 --tapir=nolo -c -emit-llvm -o /dev/null %s \
// RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s
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
// RUN: %kitcc -Oz --tapir=serial -c -emit-llvm -o /dev/null %s \
// RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s
//
// -----------------------------------------------------------------------------
//
// CHECK-NOT: PreLowerVerificationPass
// CHECK-NOT: PreLowerAnnotate
// CHECK-NOT: SerializePass
// CHECK-NOT: LoopSpawningPass
// CHECK-NOT: EmbResolveLibDeviceCallsPass
// CHECK-NOT: EmbPreparePass
// CHECK-NOT: EmbLinkLibDeviceBitcodePass
// CHECK-NOT: EmbOptimizePass
// CHECK-NOT: RecomputeKernelPropertiesPass
// CHECK-NOT: GenerateCtorsPass
// CHECK-NOT: LowerRuntimeIntrinsicsPass

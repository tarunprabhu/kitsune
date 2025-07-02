// None of the Kitsune, or Tapir, passes should run during the prelink phase,
// regardless of the specified optimization level.
//
// -----------------------------------------------------------------------------
// Only the 'none' tapir target is allowed at -O0
//
// RUN: %kitcc -O2 --tapir=none -c -emit-llvm -o /dev/null %s \
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
// CHECK-NOT: LowerMobileIntrinsics
// CHECK-NOT: StripKitsuneAddrSpace
// CHECK-NOT: LoopSpawning
// CHECK-NOT: EmbResolveLibDeviceCalls
// CHECK-NOT: EmbPreparePass
// CHECK-NOT: EmbLinkLibDeviceBitcode
// CHECK-NOT: EmbOptimize
// CHECK-NOT: RecomputeKernelProperties
// CHECK-NOT: GenerateCtors
// CHECK-NOT: LowerKitsuneRuntimeIntrinsics

// If the --tapir argument is not given, some mandatory Kitsune passes should
// run, but most should not.
//
// RUN: %kitxx -O3 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s
//
// CHECK: LowerMobileIntrinsics
// CHECK: StripKitsuneAddrSpace
// CHECK-NOT: EmbResolveLibDeviceCalls
// CHECK-NOT: EmbPreparePass
// CHECK-NOT: EmbLinkLibDeviceBitcode
// CHECK-NOT: EmbOptimize
// CHECK-NOT: RecomputeKernelProperties
// CHECK-NOT: GenerateCtors
// CHECK-NOT: LowerKitsuneRuntimeIntrinsics

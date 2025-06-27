// Most Kitsune passes should not be run during the prelink phase of LTO, but
// the mandatory passes must run.
//
// RUN: %kitcc -O3 --tapir=serial -c -emit-llvm -o /dev/null %s \
// RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
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

int main() {
  return 0;
}

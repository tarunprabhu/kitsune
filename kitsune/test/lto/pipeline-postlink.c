// The Kitsune passes should be run during the postlink phase of LTO.
//
// RUN: %kitcc -O3 --tapir=cuda -o /dev/null %s \
// RUN:     -flto -Xlinker --lto-debug-pass-manager 2>&1 \
// RUN:     | FileCheck %s
//
// CHECK: LowerMobileIntrinsics
// CHECK: StripKitsuneAddrSpace
// CHECK: EmbResolveLibDeviceCalls
// CHECK: EmbPreparePass
// CHECK: EmbLinkLibDeviceBitcode
// CHECK: EmbOptimize
// CHECK: RecomputeKernelProperties
// CHECK: GenerateCtors
// CHECK: LowerKitsuneRuntimeIntrinsics

int main() {
  return 0;
}

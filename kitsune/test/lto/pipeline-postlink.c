// The Kitsune passes should be run during the postlink phase of LTO.
//
// RUN: %kitcc -O3 --tapir=cuda -Xlinker --lto-emit-llvm -o /dev/null %s \
// RUN:     -flto -Xlinker --lto-debug-pass-manager 2>&1 \
// RUN:     | FileCheck %s
//
// CHECK: Running pass: LowerMobileIntrinsicsPass
// CHECK-NEXT: Running analysis: TapirTargetAnalysis
// CHECK-NEXT: Running analysis: LoopAnalysis
// CHECK-NEXT: Running analysis: TaskAnalysis
// CHECK-NEXT: Running pass: StripKitsuneAddrSpacePass
// CHECK: Running pass: LoopSpawningPass
// CHECK: Running pass: GlobalDCEPass
// CHECK-NEXT: Running pass: EmbResolveLibDeviceCallsPass
// CHECK-NEXT: Running pass: EmbPreparePass
// CHECK-NEXT: Running pass: EmbLinkLibDeviceBitcodePass
// CHECK-NEXT: Running pass: EmbOptimizePass
// CHECK-NEXT: Running pass: RecomputeKernelPropertiesPass
// CHECK-NEXT: Running pass: GenerateCtorsPass
// CHECK-NEXT: Running pass: LowerKitsuneRuntimeIntrinsicsPass

int main() {
  return 0;
}

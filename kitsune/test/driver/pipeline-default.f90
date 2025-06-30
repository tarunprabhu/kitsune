! REQUIRES: kitfc
!
! If the --tapir argument is provided, all Kitsune passes should run.
!
! RUN: %kitfc --tapir=serial -O3 -c -emit-llvm -o /dev/null %s \
! RUN:     -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s
!
! CHECK: Running pass: LowerMobileIntrinsicsPass
! CHECK-NEXT: Running analysis: TapirTargetAnalysis
! CHECK-NEXT: Running pass: StripKitsuneAddrSpacePass
! CHECK-NEXT: Running pass: LoopSpawningPass
! CHECK: Running pass: GlobalDCEPass
! CHECK-NEXT: Running pass: EmbResolveLibDeviceCallsPass
! CHECK-NEXT: Running pass: EmbPreparePass
! CHECK-NEXT: Running pass: EmbLinkLibDeviceBitcodePass
! CHECK-NEXT: Running pass: EmbOptimizePass
! CHECK-NEXT: Running pass: RecomputeKernelPropertiesPass
! CHECK-NEXT: Running pass: GenerateCtorsPass
! CHECK-NEXT: Running pass: LowerKitsuneRuntimeIntrinsicsPass

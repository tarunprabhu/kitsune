! REQUIRES: kitfc
!
! If the --tapir argument is not given, some mandatory Kitsune passes should
! run, but most should not.
!
! RUN: %kitfc -O3 -c -emit-llvm -o /dev/null %s \
! RUN:     -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s
!
! CHECK: LowerMobileIntrinsics
! CHECK-NEXT: Running analysis: TapirTargetAnalysis
! CHECK-NEXT: Running pass: StripKitsuneAddrSpace
! CHECK-NOT: Running pass: EmbResolveLibDeviceCallsPass
! CHECK-NOT: Running pass: EmbPreparePass
! CHECK-NOT: Running pass: EmbLinkLibDeviceBitcodePass
! CHECK-NOT: Running pass: EmbOptimizePass
! CHECK-NOT: Running pass: RecomputeKernelPropertiesPass
! CHECK-NOT: Running pass: GenerateCtorsPass
! CHECK-NOT: Running pass: LowerKitsuneRuntimeIntrinsicsPass

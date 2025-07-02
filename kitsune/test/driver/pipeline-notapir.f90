! REQUIRES: kitfc
!
! If the --tapir argument is not given, some mandatory Kitsune passes should
! run, but most should not.
!
! TODO: -Os and -Oz are not supported in flang at the time of writing these
! tests. Those will eventually be supported, at which time this test should be
! updated to include those as well.
!
! RUN: %kitfc -O0 -c -emit-llvm -o /dev/null %s \
! RUN:     -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s
!
! RUN: %kitfc -O1 -c -emit-llvm -o /dev/null %s \
! RUN:     -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s
!
! RUN: %kitfc -O2 -c -emit-llvm -o /dev/null %s \
! RUN:     -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s
!
! RUN: %kitfc -O3 -c -emit-llvm -o /dev/null %s \
! RUN:     -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s
!
! CHECK:      Running pass:     LowerMobileIntrinsics
! CHECK-NEXT: Running analysis: TapirTargetAnalysis
! CHECK-NEXT: Running pass:     StripKitsuneAddrSpacePass
! CHECK-NOT:  Running pass:     EmbResolveLibDeviceCallsPass
! CHECK-NOT:  Running pass:     EmbPreparePass
! CHECK-NOT:  Running pass:     EmbLinkLibDeviceBitcodePass
! CHECK-NOT:  Running pass:     EmbOptimizePass
! CHECK-NOT:  Running pass:     RecomputeKernelPropertiesPass
! CHECK-NOT:  Running pass:     GenerateCtorsPass
! CHECK-NEXT: Running pass:     LowerKitsuneRuntimeIntrinsicsPass

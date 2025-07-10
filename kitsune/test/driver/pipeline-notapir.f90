! REQUIRES: kitfc
!
! If the --tapir argument is not given, none of Kitsune's passes should run.
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
! CHECK-NOT:  Running analysis: TapirTargetAnalysis
! CHECK-NOT:  Running pass:     PrefetchingPass
! CHECK-NOT:  Running pass:     EmbResolveLibDeviceCallsPass
! CHECK-NOT:  Running pass:     EmbPreparePass
! CHECK-NOT:  Running pass:     EmbLinkLibDeviceBitcodePass
! CHECK-NOT:  Running pass:     EmbOptimizePass
! CHECK-NOT:  Running pass:     RecomputeKernelPropertiesPass
! CHECK-NOT:  Running pass:     GenerateCtorsPass

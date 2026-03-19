! REQUIRES: kitfc
!
! TODO: -Os and -Oz are not supported in flang at the time of writing these
! tests. Those will eventually be supported, at which time this test should be
! updated to include those as well.
!
! None of the Kitsune, or Tapir, passes should run during the prelink phase,
! regardless of the specified optimization level.
!
! -----------------------------------------------------------------------------
! Only the nolo tapir target is allowed at -O0.
!
! RUN: %kitfc -O2 --tapir=nolo -c -emit-llvm -o /dev/null %s \
! RUN:     -flto -Xflang -fdebug-pass-manager %sysroot 2>&1 \
! RUN:     | FileCheck %s
!
! -----------------------------------------------------------------------------
!
! RUN: %kitfc -O2 --tapir=serial -c -emit-llvm -o /dev/null %s \
! RUN:     -flto -Xflang -fdebug-pass-manager %sysroot 2>&1 \
! RUN:     | FileCheck %s
!
! RUN: %kitfc -O3 --tapir=serial -c -emit-llvm -o /dev/null %s \
! RUN:     -flto -Xflang -fdebug-pass-manager %sysroot 2>&1 \
! RUN:     | FileCheck %s
!
! -----------------------------------------------------------------------------
!
! CHECK-NOT: PreLowerVerificationPass
! CHECK-NOT: PreLowerAnnotate
! CHECK-NOT: SerializePass
! CHECK-NOT: LoopSpawningPass
! CHECK-NOT: EmbResolveLibDeviceCallsPass
! CHECK-NOT: EmbPreparePass
! CHECK-NOT: EmbLinkLibDeviceBitcodePass
! CHECK-NOT: EmbOptimizePass
! CHECK-NOT: RecomputeKernelPropertiesPass
! CHECK-NOT: GenerateCtorsPass
! CHECK-NOT: LowerRuntimeIntrinsicsPass

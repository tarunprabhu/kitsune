! REQUIRES: kitfc
!
! Most Kitsune passes should not be run during the prelink phase of LTO, but
! the mandatory passes must run.
!
! RUN: %kitfc -O3 --tapir=serial -c -emit-llvm -o /dev/null %s \
! RUN:     -flto -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s
!
! CHECK: LowerMobileIntrinsics
! CHECK: StripKitsuneAddrSpace
! CHECK-NOT: EmbResolveLibDeviceCalls
! CHECK-NOT: EmbPreparePass
! CHECK-NOT: EmbLinkLibDeviceBitcode
! CHECK-NOT: EmbOptimize
! CHECK-NOT: RecomputeKernelProperties
! CHECK-NOT: GenerateCtors
! CHECK-NOT: LowerKitsuneRuntimeIntrinsics

end program

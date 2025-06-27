! REQUIRES: kitfc
!
! All Kitsune passes should be run during the postlink phase of LTO.
!
! RUN: %kitfc -O3 --tapir=cuda -o /dev/null %s \
! RUN:     -flto -Xlinker --lto-debug-pass-manager 2>&1 \
! RUN:     | FileCheck %s
!
! CHECK: LowerMobileIntrinsics
! CHECK: StripKitsuneAddrSpace
! CHECK: EmbResolveLibDeviceCalls
! CHECK: EmbPreparePass
! CHECK: EmbLinkLibDeviceBitcode
! CHECK: EmbOptimize
! CHECK: RecomputeKernelProperties
! CHECK: GenerateCtors
! CHECK: LowerKitsuneRuntimeIntrinsics

end program

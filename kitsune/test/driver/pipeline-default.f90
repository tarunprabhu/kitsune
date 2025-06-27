! REQUIRES: kitfc
!
! If the --tapir argument is provided, all Kitsune passes should run.
!
! RUN: %kitfc --tapir=serial -O3 -c -emit-llvm -o /dev/null %s \
! RUN:     -Xclang -fdebug-pass-manager 2>&1 \
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

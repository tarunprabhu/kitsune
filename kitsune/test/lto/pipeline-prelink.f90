! REQUIRES: kitfc
!
! TODO: -Os and -Oz are not supported in flang at the time of writing these
! tests. Those will eventually be supported, at which time this test should be
! updated to include those as well.
!
! The Kitsune (Tapir) lowering passes should not be run during the prelink
! phase of LTO, but the non-lowering passes should be run.
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
! CHECK:      Running pass:      EarlyAnnotatePass
!
! CHECK-NOT:  Running pass:      PreLowerPreparePass
! CHECK-NOT:  Running pass:      SecondaryIVEliminationPass
! CHECK-NOT:  Running pass:      PrepareReductionLoopsPass
! CHECK-NOT:  Running pass:      LowerKitReduceIntrinsicsPass
! CHECK-NOT:  Running pass:      DeLICMPass
! CHECK-NOT:  Running pass:      PreLowerVerificationPass
! CHECK-NOT:  Running pass:      PreLowerAnnotatePass
! CHECK-NOT:  Running pass:      SerializePass
! CHECK-NOT:  Running pass:      LoopSpawningPass
! CHECK-NOT:  Running pass:      EmbResolveLibDeviceCallsPass
! CHECK-NOT:  Running pass:      EmbPreparePass
! CHECK-NOT:  Running pass:      EmbLinkLibDeviceBitcodePass
! CHECK-NOT:  Running pass:      EmbOptimizePass
! CHECK-NOT:  Running pass:      RecomputeKernelPropertiesPass
! CHECK-NOT:  Running pass:      GenerateCtorsPass
! CHECK-NOT:  Running pass:      LowerRuntimeIntrinsicsPass

subroutine f
end subroutine f

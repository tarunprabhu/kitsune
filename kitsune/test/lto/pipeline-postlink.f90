! REQUIRES: kitfc
!
! TODO: -Os and -Oz are not supported in flang at the time of writing these
! tests. Those will eventually be supported, at which time this test should be
! updated to include those as well.
!
! -----------------------------------------------------------------------------
! If the tapir target is nolo, the behavior is similar to the regular pipeline
! i.e. loop spawning is not run and neither are any Kitsune passes.
!
! RUN: %kitfc -flto -O2 --tapir=nolo -o /dev/null %s  %sysroot \
! RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
! RUN:     | FileCheck %s -check-prefix NOLO
!
! NOLO:      Running pass:     VerifierPass
! NOLO-NOT:  Running pass:     LoopSpawning
! NOLO-NOT:  Running pass:     LowerRuntimeIntrinsicsPass
! NOLO:      Running pass:     VerifierPass
! NOLO-NEXT: Running analysis: VerifierAnalysis
!
! -----------------------------------------------------------------------------
! The Kitsune (and Tapir) lowering passes should run during the postlink phase
! of LTO. But the non-lowering passes should not run.
!
! RUN: %kitfc -flto -O2 --tapir=serial -o /dev/null %s %sysroot \
! RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
! RUN:     | FileCheck %s -check-prefix O23SZ
!
! RUN: %kitfc -flto -O3 --tapir=serial -o /dev/null %s %sysroot \
! RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
! RUN:     | FileCheck %s -check-prefix O23SZ
!
! -----------------------------------------------------------------------------
!
! O23SZ-NOT:  Running pass:     EarlyVerificationPass
! O23SZ-NOT:  Running pass:     EarlyAnnotatePass
!
! O23SZ:      Running pass:     NormalizeLoopControlBlocksPass
! O23SZ:      Running pass:     SecondaryIVEliminationPass
! O23SZ:      Running pass:     PrepareTapirLoopsPass
! O23SZ:      Running pass:     LowerKitReduceIntrinsicsPass
! O23SZ:      Running pass:     DeLICMPass
! O23SZ:      Running pass:     SimplifyCFGPass
! O23SZ:      Running pass:     LoopSimplifyPass
! O23SZ:      Running pass:     PreLowerVerificationPass
! O23SZ:      Running pass:     PreLowerAnnotatePass
! O23SZ-NEXT: Running pass:     SerializePass
! O23SZ-NEXT: Running pass:     LoopSpawningPass
! O23SZ-NEXT: Running pass:     TapirToTargetPass
! O23SZ:      Running pass:     PrefetchForDevicePass
! O23SZ-NEXT: Running pass:     EmbLowerKitIntrinsicsEarlyPass
! O23SZ-NEXT: Running pass:     EmbResolveLibDeviceCallsPass
! O23SZ-NEXT: Running pass:     EmbPreparePass
! O23SZ-NEXT: Running pass:     EmbLinkLibDeviceBitcodePass
! O23SZ-NEXT: Running pass:     EmbOptimizePass
! O23SZ-NEXT: Running pass:     RecomputeKernelPropertiesPass
! O23SZ-NEXT: Running pass:     GenerateCtorsPass
! O23SZ-NEXT: Running pass:     VerifierPass
! O23SZ-NEXT: Running analysis: VerifierAnalysis

! XFAIL: *
! NOTE: This test will only work if there is a tapir loop in the body of
! subroutine f below. Since the full lowering is not yet implemented, this test
! is expected to fail. Once we implement an end-to-end lowering - say of DO
! CONCURRENT loops, this test can be re-enabled.
subroutine f
end subroutine f

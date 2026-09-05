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
! NOLO:      Running pass:     VerifierPass
! NOLO-NEXT: Running analysis: VerifierAnalysis
!
! -----------------------------------------------------------------------------
! The Kitsune (and Tapir) lowering passes should run during the postlink phase
! of LTO. But the non-lowering passes should not run.
!
! RUN: %kitfc -flto -O2 --tapir=serial -o /dev/null %s %sysroot \
! RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
! RUN:     | FileCheck %s -check-prefix O23
!
! RUN: %kitfc -flto -O3 --tapir=serial -o /dev/null %s %sysroot \
! RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
! RUN:     | FileCheck %s -check-prefix O23
!
! -----------------------------------------------------------------------------
!
! O23-NOT:    Running pass:     EarlyVerificationPass
! O23-NOT:    Running pass:     EarlyAnnotatePass
! O23-NOT:    Running pass:     PrepareTapirLoopsPass
!
! O23:        Running pass:     NormalizeLoopControlBlocksPass
! O23:        Running pass:     SecondaryIVEliminationPass
! O23:        Running pass:     DeLICMPass
! O23:        Running pass:     SimplifyCFGPass
! O23:        Running pass:     LoopSimplifyPass
! O23:        Running pass:     PreLowerVerificationPass
! O23:        Running pass:     PreLowerAnnotatePass
! O23:        Running pass:     SerializePass
! O23:        Running pass:     LoopSpawningPass
! O23:        Running pass:     HoistAllocasPass
! O23:        Running pass:     EmbHoistAllocasPass
! O23:        Running pass:     TapirToTargetPass
! O23:        Running pass:     PrefetchForDevicePass
! O23:        Running pass:     EmbFinalizeReductionsPass
! O23:        Running pass:     EmbLowerIntrinsicsPass
! O23:        Running pass:     EmbResolveLibDeviceCallsPass
! O23:        Running pass:     EmbPreparePass
! O23:        Running pass:     EmbLinkLibDeviceBitcodePass
! O23:        Running pass:     EmbOptimizePass
! O23:        Running pass:     RecomputeKernelPropertiesPass
! O23:        Running pass:     GenerateCtorsPass
! O23:        Running pass:     VerifierPass
! O23:        Running analysis: VerifierAnalysis

! XFAIL: *
! NOTE: This test will only work if there is a tapir loop in the body of
! subroutine f below. Since the full lowering is not yet implemented, this test
! is expected to fail. Once we implement an end-to-end lowering - say of DO
! CONCURRENT loops, this test can be re-enabled.
subroutine f
end subroutine f

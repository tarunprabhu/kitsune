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
! NOLO:        Running pass:       VerifierPass
! NOLO-NOT:    Running pass:       LoopSpawning
! NOLO:        Running pass:       VerifierPass
! NOLO-NEXT:   Running analysis:   VerifierAnalysis
!
! -----------------------------------------------------------------------------
! The Kitsune (and Tapir) lowering passes should run during the postlink phase
! of LTO. But the non-lowering passes should not run.
!
! RUN: %kitfc -flto -O1 --tapir=serial -o /dev/null %s %sysroot \
! RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
! RUN:     | FileCheck %s -check-prefix O123
!
! RUN: %kitfc -flto -O2 --tapir=serial -o /dev/null %s %sysroot \
! RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
! RUN:     | FileCheck %s -check-prefix O123
!
! RUN: %kitfc -flto -O3 --tapir=serial -o /dev/null %s %sysroot \
! RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
! RUN:     | FileCheck %s -check-prefix O123
!
! -----------------------------------------------------------------------------
!
! O123-NOT:    Running pass:       EarlyVerificationPass
! O123-NOT:    Running pass:       EarlyAnnotatePass
! O123-NOT:    Running pass:       SerializePass
! O123-NOT:    Running pass:       GuardLoopPass
! O123-NOT:    Running pass:       PrepareTapirLoopsPass
! O123-NOT:    Running pass:       LocalizeReductionsPass
! O123-NOT:    Running pass:       InstrumentPass
!
! ... pipeline pre-tapir
!
! ... pipeline pre-loop-spawning
! O123:        Running pass:       SecondaryIVEliminationPass
! O123:        Running pass:       PreLowerVerificationPass
! O123:        Running pass:       PreLowerAnnotatePass
! O123:        Running pass:       TapirLoopNormalizePass
!
! O123:        Running pass:       LoopSpawningPass
!
! ... pipeline post-loop-spawning
! O123:        Running pass:       HoistAllocasPass
! O123:        Running pass:       EmbHoistAllocasPass
!
! O123:        Running pass:       TapirToTargetPass
!
! ... pipeline post-tapir
! O123:        Running pass:       PrefetchForDevicePass
! O123:        Running pass:       EmbLowerIntrinsicsPass
! O123:        Running pass:       EmbResolveLibDeviceCallsPass
! O123:        Running pass:       EmbPreparePass
! O123:        Running pass:       EmbLinkLibDeviceBitcodePass
! O123:        Running pass:       EmbOptimizePass
! O123:        Running pass:       RecomputeKernelPropertiesPass
! O123:        Running pass:       GenerateCtorsPass

! O123:        Running pass:       VerifierPass
! O123:        Running analysis:   VerifierAnalysis

! XFAIL: *
! NOTE: This test will only work if there is a tapir loop in the body of
! subroutine f below. Since the full lowering is not yet implemented, this test
! is expected to fail. Once we implement an end-to-end lowering - say of DO
! CONCURRENT loops, this test can be re-enabled.
subroutine f
end subroutine f

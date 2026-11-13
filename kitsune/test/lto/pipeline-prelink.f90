! REQUIRES: kitfc
!
! TODO: -Os and -Oz are not supported in flang at the time of writing these
! tests. Those will eventually be supported, at which time this test should be
! updated to include those as well.
!
! This test is intended to check the expected passes are run in the LTO prelink
! pipeline. This requires a tapir loop to be present in the IR produced by the
! frontend. Since we don't yet have this setup for Fortran, the test will fail
! since certain passes will not run.

! XFAIL: *
!
! The Kitsune (Tapir) lowering passes should not be run during the prelink
! phase of LTO, but the non-lowering passes should be run.
!
! -----------------------------------------------------------------------------
!
! RUN: %kitfc -O1 --tapir=serial -c -emit-llvm -o /dev/null %s \
! RUN:     -flto -Xflang -fdebug-pass-manager %sysroot 2>&1 \
! RUN:     | FileCheck %s --check-prefix=O123
!
! RUN: %kitfc -O2 --tapir=serial -c -emit-llvm -o /dev/null %s \
! RUN:     -flto -Xflang -fdebug-pass-manager %sysroot 2>&1 \
! RUN:     | FileCheck %s --check-prefix=O123
!
! RUN: %kitfc -O3 --tapir=serial -c -emit-llvm -o /dev/null %s \
! RUN:     -flto -Xflang -fdebug-pass-manager %sysroot 2>&1 \
! RUN:     | FileCheck %s --check-prefix=O123
!
! -----------------------------------------------------------------------------
!
! O123:        Running pass:      EarlyVerificationPass
! O123:        Running pass:      EarlyAnnotatePass
! O123:        Running pass:      SerializePass
! O123:        Running pass:      PrepareTapirLoopsPass
! O123:        Running pass:      LocalizeReductionsPass
! O123:        Running pass:      SecondaryIVEliminationPass
! O123-NOT:    Running pass:      InstrumentPass
!
! O123-NOT:    Running pass:      SecondaryIVEliminationPass
! O123-NOT:    Running pass:      PreLowerVerificationPass
! O123-NOT:    Running pass:      PreLowerAnnotatePass
! O123-NOT:    Running pass:      TapirLoopNormalizePass
! O123-NOT:    Running pass:      LoopSpawningPass
! O123-NOT:    Running pass:      HoistAllocasPass
! O123-NOT:    Running pass:      EmbHoistAllocasPass
! O123-NOT:    Running pass:      EmbLowerIntrinsicsPass
! O123-NOT:    Running pass:      EmbResolveLibDeviceCallsPass
! O123-NOT:    Running pass:      EmbPreparePass
! O123-NOT:    Running pass:      EmbLinkLibDeviceBitcodePass
! O123-NOT:    Running pass:      EmbOptimizePass
! O123-NOT:    Running pass:      RecomputeKernelPropertiesPass
! O123-NOT:    Running pass:      GenerateCtorsPass
! O123-NOT:    Running pass:      LowerRuntimeIntrinsicsPass
!
! -----------------------------------------------------------------------------
! The instrumentation pass will only run if instrumentation is explicitly
! enabled.
!
! RUN: %kitfc -O2 --tapir=serial -c -emit-llvm -o /dev/null %s \
! RUN:     --kit-instr=generic \
! RUN:     -flto -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s --check-prefix=INSTR
!
! RUN: %kitfc -O3 --tapir=serial -c -emit-llvm -o /dev/null %s \
! RUN:     --kit-instr=timer \
! RUN:     -flto -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s --check-prefix=INSTR
!
! INSTR:      Running pass:      PrepareTapirLoopsPass
! INSTR:      Running pass:      InstrumentPass
!
! -----------------------------------------------------------------------------

subroutine f
end subroutine f

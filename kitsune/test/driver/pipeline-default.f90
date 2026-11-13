! REQUIRES: kitfc
!
! TODO: -Os and -Oz are not supported in flang at the time of writing these
! tests. Those will eventually be supported, at which time this test should be
! updated to include those as well.
!
! If the --tapir=nolo option is provided without optimizations, neither tapir,
! nor Kitsune, passes are run.
!
! RUN: %kitfc --tapir=nolo -O0 -c -emit-llvm -o /dev/null %s \
! RUN:     -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s -check-prefix O0
!
! O0-NOT:      Running pass:       SerializePass
! O0-NOT:      Running pass:       GuardLoopPass
! O0-NOT:      Running pass:       PrepareTapirLoopsPass
! O0-NOT:      Running pass:       LocalizeReductionsPass
! O0-NOT:      Running pass:       InstrumentPass
! O0-NOT:      Running pass:       SecondaryIVEliminationPass
! O0-NOT:      Running pass:       PreLowerVerificationPass
! O0-NOT:      Running pass:       PreLowerAnnotatePass
! O0-NOT:      Running pass:       TapirLoopNormalizePass
! O0-NOT:      Running pass:       LoopSpawningPass
! O0-NOT:      Running pass:       TapirToTargetPass
! O0-NOT:      Running pass:       HoistAllocasPass
! O0-NOT:      Running pass:       EmbHoistAllocasPass
! O0-NOT:      Running pass:       PrefetchForDevicePass
! O0-NOT:      Running pass:       EmbLowerIntrinsicsPass
! O0-NOT:      Running pass:       EmbResolveLibDeviceCallsPass
! O0-NOT:      Running pass:       EmbPreparePass
! O0-NOT:      Running pass:       EmbLinkLibDeviceBitcodePass
! O0-NOT:      Running pass:       EmbOptimizePass
! O0-NOT:      Running pass:       RecomputeKernelPropertiesPass
! O0-NOT:      Running pass:       GenerateCtorsPass
!
! -----------------------------------------------------------------------------
! If the --tapir argument is provided, all Tapir and Kitsune passes should run.
!
! RUN: %kitfc --tapir=serial -O1 -c -emit-llvm -o /dev/null %s \
! RUN:     -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s -check-prefix O123
!
! RUN: %kitfc --tapir=serial -O2 -c -emit-llvm -o /dev/null %s \
! RUN:     -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s -check-prefix O123
!
! RUN: %kitfc --tapir=serial -O3 -c -emit-llvm -o /dev/null %s \
! RUN:     -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s -check-prefix O123
!
! ... pipeline early
! O123:        Running pass:       EarlyVerificationPass
! O123:        Running pass:       EarlyAnnotatePass
!
! ... pipeline prepare
! O123:        Running pass:       SerializePass
! O123:        Running pass:       GuardLoopPass
! O123:        Running pass:       PrepareTapirLoopsPass
! O123:        Running pass:       LocalizeReductionsPass
! O123S-NOT:   Running pass:       InstrumentPass
!
! ... pipeline pre-vectorize
! O123:        Running pass:       SecondaryIVEliminationPass
!
! ... pipeline pre-tapir
! There are no standard pre-tapir passes at this time
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
!
! O123:        Running pass:       BitcodeWriterPass
!
! ------------------------------------------------------------------------------
! The instrumentation pass will only run if instrumentation is explicitly
! enabled.
!
! RUN: %kitfc --tapir=serial -O1 -c -emit-llvm -o /dev/null %s \
! RUN:     --kit-instr=timer \
! RUN:     -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s -check-prefix INSTR
!
! RUN: %kitfc --tapir=serial -O2 -c -emit-llvm -o /dev/null %s \
! RUN:     --kit-instr=generic \
! RUN:     -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s -check-prefix INSTR
!
! RUN: %kitfc --tapir=serial -O3 -c -emit-llvm -o /dev/null %s \
! RUN:     --kit-instr=generic,timer \
! RUN:     -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s -check-prefix INSTR
!
! INSTR:      Running pass:     PrepareTapirLoopsPass
! INSTR:      Running pass:     InstrumentPass
!
! ------------------------------------------------------------------------------

! XFAIL: *
! NOTE: This test will only work if there is a tapir loop in the body of
! subroutine f below. Since the full lowering is not yet implemented, this test
! is expected to fail. Once we implement an end-to-end lowering - say of DO
! CONCURRENT loops, this test can be re-enabled.
subroutine f()
end subroutine f

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
! O0-NOT: Running pass: PreLowerAnnotate
! O0-NOT: Running pass: LoopSpawningPass
!
! -----------------------------------------------------------------------------
! If the --tapir argument is provided, all Tapir and Kitsune passes should run.
!
! RUN: %kitfc --tapir=serial -O1 -c -emit-llvm -o /dev/null %s \
! RUN:     -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s -check-prefix O123S
!
! RUN: %kitfc --tapir=serial -O2 -c -emit-llvm -o /dev/null %s \
! RUN:     -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s -check-prefix O123S
!
! RUN: %kitfc --tapir=serial -O3 -c -emit-llvm -o /dev/null %s \
! RUN:     -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s -check-prefix O123S
!
! The EarlyAnnotatePass runs early in the pass pipeline.
! O123S:      Running pass:     EarlyAnnotatePass
!
! <KIT-PRE-TAPIR>
! There are no standard pre-tapir passes at this time
! </KIT-PRE-TAPIR>
!
! <KIT-PRE-LOOP-SPAWNING>
! We add LoopSimplify, LoopRotate and LoopLCSSA to the pipeline before
! PrepareReductionLoops, but it is difficult to check for them because they
! match runs of the pass from earlier in the pipeline. PrepareReductionLoops
! will fail if any of these are not run, so something will at least catch it
! if they are ever removed from the pipeline.
! O123S:      Running pass:     SecondaryIVEliminationPass
! O123S:      Running pass:     PrepareReductionLoopsPass
! O123S:      Running pass:     LowerKitReduceIntrinsicsPass
! O123S:      Running pass:     ModuleInlinerPass
! O123S:      Running pass:     EarlyCSEPass
! O123S:      Running pass:     SimplifyCFGPass
! O123S:      Running pass:     InstCombinePass
! O123S:      Running pass:     SCCPPass
! O123S:      Running pass:     BDCEPass
! O123S:      Running pass:     InstCombinePass
! O123S:      Running pass:     DSEPass
! O123S:      Running pass:     ADCEPass
! O123S:      Running pass:     DeLICMPass
! O123S:      Running pass:     SimplifyCFGPass
! O123S:      Running pass:     LoopSimplifyPass
! O123S:      Running pass:     PreLowerVerificationPass
! O123S:      Running pass:     PreLowerAnnotate
! O123S:      Running pass:     SerializePass
! </KIT-PRE-LOOP-SPAWNING>
!
! O123S-NEXT: Running pass:     LoopSpawningPass
! O123S:      Running pass:     TapirToTargetPass
! O123S:      Running pass:     GlobalDCEPass
!
! <KIT-POST-TAPIR>
! O123S:      Running pass:     PrefetchForDevicePass
! O123S:      Running pass:     EmbLowerKitIntrinsicsEarlyPass
! O123S:      Running pass:     EmbResolveLibDeviceCallsPass
! O123S:      Running pass:     EmbPreparePass
! O123S:      Running pass:     EmbLinkLibDeviceBitcodePass
! O123S:      Running pass:     EmbOptimizePass
! O123S:      Running pass:     RecomputeKernelPropertiesPass
! O123S:      Running pass:     GenerateCtorsPass
! </KIT-POST-TAPIR>
!
! O123S:      Running pass:     BitcodeWriterPass

subroutine f()
end subroutine f

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
! RUN:     | FileCheck %s -check-prefix O123SZ
!
! RUN: %kitfc --tapir=serial -O2 -c -emit-llvm -o /dev/null %s \
! RUN:     -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s -check-prefix O123SZ
!
! RUN: %kitfc --tapir=serial -O3 -c -emit-llvm -o /dev/null %s \
! RUN:     -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s -check-prefix O123SZ
!
! The EarlyAnnotatePass runs early in the pass pipeline.
! O123SZ:      Running pass:     EarlyAnnotatePass
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
! O123SZ:      Running pass:     PrepareReductionLoopsPass
! O123SZ:      Running pass:     LowerKitReduceIntrinsicsPass
! O123SZ:      Running pass:     ModuleInlinerPass
! O123SZ:      Running pass:     EarlyCSEPass
! O123SZ:      Running pass:     SimplifyCFGPass
! O123SZ:      Running pass:     InstCombinePass
! O123SZ:      Running pass:     SCCPPass
! O123SZ:      Running pass:     BDCEPass
! O123SZ:      Running pass:     InstCombinePass
! O123SZ:      Running pass:     DSEPass
! O123SZ:      Running pass:     ADCEPass
! O123SZ:      Running pass:     DeLICMPass
! O123SZ:      Running pass:     SimplifyCFGPass
! O123SZ:      Running pass:     LoopSimplifyPass
! O123SZ:      Running pass:     PreLowerVerificationPass
! O123SZ:      Running pass:     PreLowerAnnotate
! O123SZ:      Running pass:     SerializePass
! </KIT-PRE-LOOP-SPAWNING>
!
! O123SZ-NEXT: Running pass:     LoopSpawningPass
! O123SZ:      Running pass:     TapirToTargetPass
! O123SZ:      Running pass:     GlobalDCEPass
!
! <KIT-POST-TAPIR>
! O123SZ:      Running pass:     PrefetchForDevicePass
! O123SZ:      Running pass:     EmbLowerKitIntrinsicsEarlyPass
! O123SZ:      Running pass:     EmbResolveLibDeviceCallsPass
! O123SZ:      Running pass:     EmbPreparePass
! O123SZ:      Running pass:     EmbLinkLibDeviceBitcodePass
! O123SZ:      Running pass:     EmbOptimizePass
! O123SZ:      Running pass:     RecomputeKernelPropertiesPass
! O123SZ:      Running pass:     GenerateCtorsPass
! </KIT-POST-TAPIR>
!
! O123SZ:      Running pass:     BitcodeWriterPass

subroutine f()
end subroutine f

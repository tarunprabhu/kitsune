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
! O123SZ:      Running pass:     EarlyAnnotatePass
! O123SZ:      Running pass:     AnnotationRemarksPass
! O123SZ-NEXT: Running pass:     LoopSimplifyPass
! O123SZ-NEXT: Running analysis: LoopAnalysis
! O123SZ-NEXT: Running analysis: DominatorTreeAnalysis
! O123SZ:      Running pass:     PreLowerVerificationPass
! O123SZ-NEXT: Running analysis: TTObjectsAnalysis
! O123SZ-NEXT: Running analysis: TaskAnalysis
! O123SZ-NEXT: Running analysis: PostDominatorTreeAnalysis
! O123SZ-NEXT: Running analysis: ScalarEvolutionAnalysis
! O123SZ-NEXT: Running pass:     PreLowerAnnotate
! O123SZ-NEXT: Running pass:     SerializePass
! O123SZ-NEXT: Running pass:     LoopSpawningPass
! O123SZ:      Running pass:     TapirToTargetPass
! O123SZ:      Running pass:     GlobalDCEPass
! O123SZ-NEXT: Running pass:     PrefetchForDevicePass
! O123SZ-NEXT: Running pass:     EmbLowerKitIntrinsicsLibDevicePass
! O123SZ-NEXT: Running pass:     EmbResolveLibDeviceCallsPass
! O123SZ-NEXT: Running pass:     EmbPreparePass
! O123SZ-NEXT: Running pass:     EmbLinkLibDeviceBitcodePass
! O123SZ-NEXT: Running pass:     EmbOptimizePass
! O123SZ-NEXT: Running pass:     RecomputeKernelPropertiesPass
! O123SZ-NEXT: Running pass:     GenerateCtorsPass
! O123SZ-NEXT: Running pass:     BitcodeWriterPass

subroutine f()
end subroutine f

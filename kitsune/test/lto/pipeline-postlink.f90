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
! RUN: %kitfc -O2 --tapir=nolo -o /dev/null %s  %sysroot \
! RUN:     -flto -Xlinker --lto-debug-pass-manager 2>&1 \
! RUN:     | FileCheck %s -check-prefix NOLO
!
! NOLO:      Running pass:     VerifierPass
! NOLO-NOT:  Running pass:     LoopSpawning
! NOLO-NOT:  Running pass:     LowerRuntimeIntrinsicsPass
! NOLO:      Running pass:     VerifierPass
! NOLO-NEXT: Running analysis: VerifierAnalysis
!
! -----------------------------------------------------------------------------
! All Kitsune, and Tapir, passes should run during the postlink phase of LTO.
!
! RUN: %kitfc -O2 --tapir=serial -o /dev/null %s %sysroot \
! RUN:     -flto -Xlinker --lto-debug-pass-manager 2>&1 \
! RUN:     | FileCheck %s -check-prefix O23SZ
!
! RUN: %kitfc -O3 --tapir=serial -o /dev/null %s %sysroot \
! RUN:     -flto -Xlinker --lto-debug-pass-manager 2>&1 \
! RUN:     | FileCheck %s -check-prefix O23SZ
!
! -----------------------------------------------------------------------------
!
! O23SZ:      Running pass:     ReversePostOrderFunctionAttrs
! O23SZ:      Running pass:     LoopSpawningPass
! O23SZ:      Running pass:     TapirToTargetPass
! O23SZ:      Running pass:     ReversePostOrderFunctionAttrs
! O23SZ-NEXT: Running pass:     GlobalDCEPass
! O23SZ-NEXT: Running pass:     PrefetchingPass
! O23SZ-NEXT: Running analysis: TapirTargetAnalysis
! O23SZ-NEXT: Running pass:     EmbResolveLibDeviceCallsPass
! O23SZ-NEXT: Running pass:     EmbPreparePass
! O23SZ-NEXT: Running pass:     EmbLinkLibDeviceBitcodePass
! O23SZ-NEXT: Running pass:     EmbOptimizePass
! O23SZ-NEXT: Running pass:     RecomputeKernelPropertiesPass
! O23SZ-NEXT: Running pass:     GenerateCtorsPass
! O23SZ-NEXT: Running pass:     AnnotationRemarksPass
! O23SZ-NEXT: Running pass:     VerifierPass
! O23SZ-NEXT: Running analysis: VerifierAnalysis

end program

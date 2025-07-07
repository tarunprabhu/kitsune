! REQUIRES: kitfc
!
! TODO: -Os and -Oz are not supported in flang at the time of writing these
! tests. Those will eventually be supported, at which time this test should be
! updated to include those as well.
!
! -----------------------------------------------------------------------------
! If the tapir target is 'none', the behavior is similar to the regular
! pipeline i.e. loop spawning is not run and neither are any Kitsune passes.
!
! RUN: %kitfc -O2 --tapir=none -Xlinker --lto-emit-llvm -o /dev/null %s \
! RUN:     -flto -Xlinker --lto-debug-pass-manager %sysroot 2>&1 \
! RUN:     | FileCheck %s -check-prefix NONE
!
! NONE:      Running pass:     VerifierPass
! NONE-NOT:  Running pass:     LoopSpawning
! NONE-NOT:  Running pass:     LowerRuntimeIntrinsicsPass
! NONE:      Running pass:     VerifierPass
! NONE-NEXT: Running analysis: VerifierAnalysis
!
! -----------------------------------------------------------------------------
! All Kitsune, and Tapir, passes should run during the postlink phase of LTO.
!
! RUN: %kitfc -O2 --tapir=serial -Xlinker --lto-emit-llvm -o /dev/null %s \
! RUN:     -flto -Xlinker --lto-debug-pass-manager %sysroot 2>&1 \
! RUN:     | FileCheck %s -check-prefix O23SZ
!
! RUN: %kitfc -O3 --tapir=serial -Xlinker --lto-emit-llvm -o /dev/null %s \
! RUN:     -flto -Xlinker --lto-debug-pass-manager %sysroot 2>&1 \
! RUN:     | FileCheck %s -check-prefix O23SZ
!
! -----------------------------------------------------------------------------
!
! O23SZ:      Running pass:     GlobalDCEPass
! O23SZ:      Running pass:     LoopSpawningPass
! O23SZ-NEXT: Running analysis: TapirTargetAnalysis
! O23SZ-NEXT: Running pass:     TapirToTargetPass
! O23SZ-NEXT: Running pass:     IPSCCPPass
! O23SZ-NEXT: Running pass:     CalledValuePropagationPass
! O23SZ-NEXT: Running pass:     GlobalOptPass
! O23SZ-NEXT: Running pass:     DeadArgumentEliminationPass
! O23SZ-NEXT: Running pass:     AlwaysInlinerPass
! O23SZ-NEXT: Running pass:     RequireAnalysisPass
! O23SZ-NEXT: Running pass:     EliminateAvailableExternallyPass
! O23SZ-NEXT: Running pass:     ReversePostOrderFunctionAttrs
! O23SZ-NEXT: Running pass:     GlobalDCEPass
! O23SZ-NEXT: Running pass:     EmbResolveLibDeviceCallsPass
! O23SZ-NEXT: Running pass:     EmbPreparePass
! O23SZ-NEXT: Running pass:     EmbLinkLibDeviceBitcodePass
! O23SZ-NEXT: Running pass:     EmbOptimizePass
! O23SZ-NEXT: Running pass:     RecomputeKernelPropertiesPass
! O23SZ-NEXT: Running pass:     GenerateCtorsPass
! O23SZ-NEXT: Running pass:     VerifierPass
! O23SZ-NEXT: Running analysis: VerifierAnalysis

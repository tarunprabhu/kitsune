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
! O0:      Running pass:     AlwaysInlinerPass
! O0-NEXT: Running analysis: ProfileSummaryAnalysis
! O0-NEXT: Running pass:     CoroConditionalWrapper
! O0-NEXT: Running pass:     BitcodeWriterPass
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
! O123SZ:      Running pass:     LoopSpawningPass
! O123SZ-NEXT: Running analysis: TapirTargetAnalysis
! O123SZ-NEXT: Running pass:     TapirToTargetPass
! O123SZ-NEXT: Running pass:     IPSCCPPass
! O123SZ-NEXT: Running pass:     CalledValuePropagationPass
! O123SZ-NEXT: Running pass:     GlobalOptPass
! O123SZ-NEXT: Running pass:     DeadArgumentEliminationPass
! O123SZ-NEXT: Running pass:     AlwaysInlinerPass
! O123SZ-NEXT: Running pass:     RequireAnalysisPass
! O123SZ-NEXT: Running pass:     EliminateAvailableExternallyPass
! O123SZ-NEXT: Running pass:     ReversePostOrderFunctionAttrs
! O123SZ-NEXT: Running pass:     GlobalDCEPass
! O123SZ-NEXT: Running pass:     EmbResolveLibDeviceCallsPass
! O123SZ-NEXT: Running pass:     EmbPreparePass
! O123SZ-NEXT: Running pass:     EmbLinkLibDeviceBitcodePass
! O123SZ-NEXT: Running pass:     EmbOptimizePass
! O123SZ-NEXT: Running pass:     RecomputeKernelPropertiesPass
! O123SZ-NEXT: Running pass:     GenerateCtorsPass
! O123SZ-NEXT: Running pass:     BitcodeWriterPass

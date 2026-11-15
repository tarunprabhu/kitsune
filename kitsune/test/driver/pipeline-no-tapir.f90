! REQUIRES: kitfc
!
! If the --tapir argument is not given, none of Kitsune's passes should run.
!
! TODO: -Os and -Oz are not supported in flang at the time of writing these
! tests. Those will eventually be supported, at which time this test should be
! updated to include those as well.
!
! RUN: %kitfc -O0 -c -emit-llvm -o /dev/null %s \
! RUN:     -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s
!
! RUN: %kitfc -O1 -c -emit-llvm -o /dev/null %s \
! RUN:     -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s
!
! RUN: %kitfc -O2 -c -emit-llvm -o /dev/null %s \
! RUN:     -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s
!
! RUN: %kitfc -O3 -c -emit-llvm -o /dev/null %s \
! RUN:     -Xflang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s
!
! CHECK-NOT:   Running pass:       EarlyVerificationPass
! CHECK-NOT:   Running pass:       AnnotateEarlyPass
! CHECK-NOT:   Running pass:       SerializePass
! CHECK-NOT:   Running pass:       PrepareTapirLoopsPass
! CHECK-NOT:   Running pass:       LocalizeReductionsPass
! CHECK-NOT:   Running pass:       LowerIntrinsicsEarlyPass
! CHECK-NOT:   Running pass:       InstrumentPass
! CHECK-NOT:   Running pass:       SecondaryIVEliminationPass
! CHECK-NOT:   Running pass:       SecondaryIVEliminationPass
! CHECK-NOT:   Running pass:       PreLowerVerificationPass
! CHECK-NOT:   Running pass:       AnnotatePreLowerPass
! CHECK-NOT:   Running pass:       TapirLoopNormalizePass
! CHECK-NOT:   Running pass:       LoopSpawningPass
! CHECK-NOT:   Running pass:       HoistAllocasPass
! CHECK-NOT:   Running pass:       EmbHoistAllocasPass
! CHECK-NOT:   Running pass:       TapirToTargetPass
! CHECK-NOT:   Running pass:       PrefetchForDevicePass
! CHECK-NOT:   Running pass:       EmbLowerIntrinsicsPass
! CHECK-NOT:   Running pass:       EmbResolveLibDeviceCallsPass
! CHECK-NOT:   Running pass:       EmbPreparePass
! CHECK-NOT:   Running pass:       EmbLinkLibDeviceBitcodePass
! CHECK-NOT:   Running pass:       EmbOptimizePass
! CHECK-NOT:   Running pass:       RecomputeKernelPropertiesPass
! CHECK-NOT:   Running pass:       GenerateCtorsPass

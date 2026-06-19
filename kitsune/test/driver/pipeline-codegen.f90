! REQUIRES: kitfc
!
! TODO: -Os and -Oz are not supported in flang at the time of writing these
! tests. Those will eventually be supported, at which time this test should be
! updated to include those as well.
!
! -----------------------------------------------------------------------------
! If the --tapir option is not provided, the Kitsune passes are not run.
!
! RUN: %kitfc -O0 -mllvm -debug-pass=Structure %s -c -o /dev/null 2>&1 \
! RUN:     | FileCheck -check-prefix DEFAULT %s
!
! RUN: %kitfc -O1 -mllvm -debug-pass=Structure %s -c -o /dev/null 2>&1 \
! RUN:     | FileCheck -check-prefix DEFAULT %s
!
! RUN: %kitfc -O2 -mllvm -debug-pass=Structure %s -c -o /dev/null 2>&1 \
! RUN:     | FileCheck -check-prefix DEFAULT %s
!
! RUN: %kitfc -O3 -mllvm -debug-pass=Structure %s -c -o /dev/null 2>&1 \
! RUN:     | FileCheck -check-prefix DEFAULT %s
!
! DEFAULT-NOT: Strip Kitsune address spaces
! DEFAULT-NOT: Lower Kitsune intrinsics
! DEFAULT-NOT: Generate Kitsune fat binaries
!
! -----------------------------------------------------------------------------
! If the --tapir option is provided, the Kitsune passes are run at all
! optimization levels.
!
! RUN: %kitfc -O0 --tapir=nolo %s -c -o /dev/null \
! RUN:     -mllvm -debug-pass=Structure 2>&1 \
! RUN:     | FileCheck %s -check-prefix TAPIR
!
! RUN: %kitfc -O1 --tapir=nolo %s -c -o /dev/null \
! RUN:     -mllvm -debug-pass=Structure 2>&1 \
! RUN:     | FileCheck %s -check-prefix TAPIR
!
! RUN: %kitfc -O2 --tapir=nolo %s -c -o /dev/null \
! RUN:     -mllvm -debug-pass=Structure 2>&1 \
! RUN:     | FileCheck %s -check-prefix TAPIR
!
! RUN: %kitfc -O3 --tapir=nolo %s -c -o /dev/null \
! RUN:     -mllvm -debug-pass=Structure 2>&1 \
! RUN:     | FileCheck %s -check-prefix TAPIR
!
! TAPIR: ModulePass Manager
! TAPIR-NEXT: Lower Kitsune intrinsics (embedded)
! TAPIR: FunctionPass Manager
! TAPIR-NEXT: Lower Kitsune intrinsics
! TAPIR-NEXT: Strip Kitsune address spaces
! TAPIR-NEXT: Generate Kitsune fat binaries
! TAPIR-NEXT: Pre-ISel Intrinsic Lowering
!
! -----------------------------------------------------------------------------

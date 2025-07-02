! REQUIRES: kitfc
!
! TODO: -Os and -Oz are not supported in flang at the time of writing these
! tests. Those will eventually be supported, at which time this test should be
! updated to include those as well.
!
! -----------------------------------------------------------------------------
! The --tapir option requires optimizations, unless the tapir target is none.
!
! RUN: not %kitfc --tapir=serial %s -c -emit-llvm -o /dev/null 2>&1 \
! RUN:      | FileCheck %s --check-prefix=O1
!
! RUN: not %kitfc --tapir=serial -O0 %s -c -emit-llvm -o /dev/null 2>&1 \
! RUN:      | FileCheck %s --check-prefix=O1
!
! -----------------------------------------------------------------------------
!
! RUN: %kitfc --tapir=none -O0 %s -c -emit-llvm -o /dev/null 2>&1 \
! RUN:      | FileCheck %s --allow-empty --check-prefix=OK
!
! -----------------------------------------------------------------------------
! Sanity check that we don't *always* require optimizations.
!
! RUN: %kitfc %s -c -emit-llvm -o /dev/null \
! RUN:     | FileCheck %s --allow-empty -check-prefix OK
!
! RUN: %kitfc -O0 %s -c -emit-llvm -o /dev/null \
! RUN:     | FileCheck %s --allow-empty -check-prefix OK
!
! -----------------------------------------------------------------------------
! If -flto is given, at least O2 is required. There is true even if the tapir
! target is set to 'none'
!
! RUN: not %kitfc -flto --tapir=serial -O0 %s -c -emit-llvm -o /dev/null 2>&1 \
! RUN:      | FileCheck %s --check-prefix=O2
!
! RUN: not %kitfc -flto --tapir=serial -O1 %s -c -emit-llvm -o /dev/null 2>&1 \
! RUN:      | FileCheck %s --check-prefix=O2
!
! RUN: not %kitfc -flto --tapir=none -O1 %s -c -emit-llvm -o /dev/null 2>&1 \
! RUN:      | FileCheck %s --check-prefix=O2
!
! RUN: %kitfc -flto --tapir=serial -O2 %s -c -o /dev/null 2>&1 \
! RUN:      | FileCheck %s --allow-empty --check-prefix=OK
!
! RUN: %kitfc -flto --tapir=serial -O3 %s -c -o /dev/null 2>&1 \
! RUN:      | FileCheck %s --allow-empty --check-prefix=OK
!
! -----------------------------------------------------------------------------
!
! O1: error: --tapir requires optimization level O1 or higher
! O2: error: --tapir requires optimization level O2 or higher for LTO
! OK-NOT: {{.+}}

end program

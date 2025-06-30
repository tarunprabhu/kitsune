! REQUIRES: kitfc
!
! -----------------------------------------------------------------------------
! The --tapir option requires optimizations, unless the tapir target is none.
!
! RUN: not %kitfc --tapir=serial %s -c -emit-llvm -o /dev/null 2>&1 \
! RUN:      | FileCheck %s --check-prefix=ERROR
!
! RUN: not %kitfc --tapir=serial -O0 %s -c -emit-llvm -o /dev/null 2>&1 \
! RUN:      | FileCheck %s --check-prefix=ERROR
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
!
! ERROR: error: --tapir requires optimization level O1 or higher
! OK-NOT: {{.+}}

end program

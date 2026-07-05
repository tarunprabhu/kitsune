! REQUIRES: kitfc
!
! -fkokkos cannot be used with Kitsune's Fortran driver.
!
! RUN: not %kitfc -### --kokkos %s 2>&1 | FileCheck %s
! RUN: not %kitfc -### --kokkos-no-init %s 2>&1 | FileCheck %s
!
! CHECK: option '--kokkos{{.*}}' can only be used with kit++

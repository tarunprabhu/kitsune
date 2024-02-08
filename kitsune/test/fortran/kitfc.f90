! Check that the Kitsune Fortran frontend does not link any kitsune runtime
! libraries if a -ftapir flag is not been specified.
!
! A default tapir target should *not* be added.

! RUN: %kitfc -### %s 2>&1 | FileCheck %s

! CHECK-NOT: -ftapir
! CHECK-NOT: -lkit{{.+}}

end program

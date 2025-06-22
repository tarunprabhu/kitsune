! REQUIRES: kitfc
!
! This has not been checked, so force it to fail if we ever resurrect this
! tapir target just so we are forced to take a look at this.
!
! RUN: false
!
! ------------------------------------------------------------------------------
! RUN: %kitfc -### -ftapir=qthreads %s 2>&1 | FileCheck %s
! RUN: %kitfc -### --tapir=qthreads %s 2>&1 | FileCheck %s
!
! CHECK: -fc1
! CHECK-SAME: --tapir=qthreads
!
! It is a pain to check for the actual linker executable. There are far too
! many options depending on the platform, so just check the next line for the
! expected linker flags.
!
! CHECK-NEXT: -lqthreads
! CHECK-SAME: -lkitrt

end program

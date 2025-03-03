! REQUIRES: kitfc
!
! ------------------------------------------------------------------------------
! RUN: %kitfc -### -ftapir=serial -O2 %s 2>&1 | FileCheck %s
! RUN: %kitfc -### --tapir=serial -O2 %s 2>&1 | FileCheck %s
!
! CHECK: -fc1
! CHECK-SAME: --tapir=serial
! CHECK-SAME: -fstripmine
!
! It is a pain to check for the actual linker executable. There are far too
! many options depending on the platform, so just check the next line for the
! expected linker flags.
!
! CHECK-NEXT: -lkitrt

end program

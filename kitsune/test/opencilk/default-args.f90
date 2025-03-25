! REQUIRES: kitfc
!
! Check that the default options added to the internal command lines (for -fc1
! and the linker) are as expected.
!
! -print-pipeline-passes currently does not work with flang, but has been
! implemented upstream. This should pass once we merge with upstream.
! XFAIL: *
!
! ------------------------------------------------------------------------------
! RUN: %kitfc -### -ftapir=opencilk -O2 %s 2>&1 | FileCheck %s
! RUN: %kitfc -### --tapir=opencilk -O2 %s 2>&1 | FileCheck %s
!
! CHECK: -fc1
! CHECK-SAME: --tapir=opencilk
! CHECK-SAME: --tapir-opencilk-abi-bc
! CHECK-SAME: -fstripmine
!
! It is a pain to check for the actual linker executable. There are far too
! many options depending on the platform, so just check the next line for the
! expected linker flags.
!
! CHECK-NEXT: -lopencilk
! CHECK-SAME: -lkitrt

end program

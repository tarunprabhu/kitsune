! REQUIRES: kitfc
!
! ------------------------------------------------------------------------------
! Check that an error is emitted if any of the required options are not
! provided
!
! RUN: not %kitfc -fc1 --tapir=opencilk %s -o /dev/null 2>&1 \
! RUN:     | FileCheck %s -check-prefix=MISSING_RUNTIME_BC
!
! MISSING_RUNTIME_BC: missing required option '--tapir-opencilk-runtime-bc='
!
! ------------------------------------------------------------------------------

end program

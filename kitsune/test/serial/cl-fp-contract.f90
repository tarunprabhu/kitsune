! REQUIRES: kitfc
!
! Check that the -ffp-contract option is handled correctly since our handling
! of this option is slightly different from flang's.
!
! RUN: %kitfc -fc1 -emit-mlir -o - %s -ffp-contract=off --tapir=serial 2>&1 \
! RUN:     | FileCheck %s --check-prefix CONTRACT-OFF
!
! RUN: %kitfc -fc1 -emit-mlir -o - %s -ffp-contract=fast --tapir=serial 2>&1 \
! RUN:     | FileCheck %s --check-prefix CONTRACT-FAST
!
! CONTRACT-OFF-NOT: fastmath<contract>
! CONTRACT-FAST: fastmath<contract>

end program

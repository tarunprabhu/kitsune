! REQUIRES: kitfc
!
! Check that any explicitly specified options are handled correctly.
!
! RUN: %kitfc -fc1 -emit-mlir -o - %s -ffp-contract=off -ftapir=serial 2>&1 \
! RUN:     | FileCheck %s --check-prefix CONTRACT-OFF
!
! RUN: %kitfc -fc1 -emit-mlir -o - %s -ffp-contract=fast -ftapir=serial 2>&1 \
! RUN:     | FileCheck %s --check-prefix CONTRACT-FAST
!
! CONTRACT-OFF-NOT: fastmath<contract>
! CONTRACT-FAST: fastmath<contract>

end program

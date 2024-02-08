! REQUIRES: kitfc
!
! ------------------------------------------------------------------------------
! Unlike the other Kitsune frontends, the default FP contract value is the same
! as flang, but that default is not the same as clang's.
!
! RUN: %flang -fc1 -emit-mlir -o - %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix CONTRACT-FAST
!
! ------------------------------------------------------------------------------
! Check that kitfc uses the same defaults as flang.
!
! RUN: %kitfc -fc1 -emit-mlir -o - %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix CONTRACT-FAST
!
! CONTRACT-FAST: fastmath<contract>
!
! ------------------------------------------------------------------------------
! -ffp-contract=on is not supported in flang and maps to -ffp-contract=off. But
! we do not allow it at all in Kitsune. -ffp-contract=fast-honor-pragmas is not
! supported by Kitsune in Fortran.
!
! RUN: %flang -### -ffp-contract=on %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix ON_TO_OFF
!
! RUN: not %kitfc -### -ffp-contract=on %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix ERROR
!
! RUN: not %kitfc -### -ffp-contract=fast-honor-pragmas %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix ERROR
!
! ON_TO_OFF: warning: the argument 'on' is not supported for option 'ffp-contract='. Mapping to 'ffp-contract=off'
! ERROR: error: unsupported argument '{{.+}}' to option '-ffp-contract=' for frontend '{{.+}}'

end program

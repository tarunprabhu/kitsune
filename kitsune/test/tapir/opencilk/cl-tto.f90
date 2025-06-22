! REQUIRES: kitfc
!
! -----------------------------------------------------------------------------
! Check that the options provided to kit++ make it to the tapir target.
! Right now, there are no command line options specific to the opencilk tapir
! target.
!
! -----------------------------------------------------------------------------
! Check that the options only allowed in -cc1 make it to the tapir targets.
!
! RUN: %kitfc -fc1 --tapir=opencilk --tapir-verbose -O2 %s -o /dev/null \
! RUN:     -emit-llvm \
! RUN:     --tapir-opencilk-runtime-bc="%S/input/libopencilk-abi.bc" 2>&1 \
! RUN:     | FileCheck %s --check-prefixes ALL,FC1
!
! -----------------------------------------------------------------------------
! ALL: 'opencilk' tapir target options
! FC1:       Opencilk bitcode file: {{.+}}/libopencilk-abi.bc

end program

! REQUIRES: kitfc
!
! Check that the options required by -fc1 make it to the tapir targets.
!
! RUN: %kitfc -fc1 --tapir=opencilk --tapir-verbose -O2 %s -o /dev/null \
! RUN:     -emit-llvm \
! RUN:     --tapir-opencilk-runtime-bc="%S/input/libopencilk-abi.bc" 2>&1 \
! RUN:     | FileCheck %s
!
! CHECK: 'opencilk' tapir target options
! CHECK: Bitcode file: {{.+}}.bc

end program

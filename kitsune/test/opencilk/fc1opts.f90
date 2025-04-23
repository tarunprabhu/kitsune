! REQUIRES: kitfc
!
! XFAIL: *
! FIXME: This has not been implemented
!
! Check that the options required by -fc1 make it to the tapir targets.
!
! RUN: %kitfc -fc1 --tapir=opencilk --tapir-verbose -O2 %s -o /dev/null \
! RUN:     --tapir-opencilk-runtime-bc="%S/input/nvptx.bc" 2>&1 \
! RUN:     | FileCheck %s
!
! CHECK: 'opencilk' tapir target options
! CHECK: Bitcode file: {{.+}}.bc
!
! FIXME: We need a DO CONCURRENT loop so OpenCilkABI is entered.

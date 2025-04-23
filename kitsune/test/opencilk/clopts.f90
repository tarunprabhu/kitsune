! REQUIRES: kitfc
!
! This has not been implemented yet.
! XFAIL: *
!
! Check that the frontend options make it to the tapir target.
!
! RUN: %kitfc --tapir=opencilk --tapir-verbose \
! RUN:     -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
! RUN:     | FileCheck %s -check-prefixes ALL,COMPILE
!
! RUN: %kitfc --tapir=opencilk --tapir-verbose --kitrt-verbose \
! RUN:     -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
! RUN:     | FileCheck %s -check-prefixes ALL,RUNTIME
!
! ALL: 'opencilk' tapir target options
! COMPILE:   Runtime verbose: true
! RUNTIME:   Runtime verbose: true
! ALL:       Bitcode file: {{.+}}/libopencilk-abi.bc

! FIXME: We need a DO CONCURRENT loop to ensure OpenCilkABI is entered

! REQUIRES: kitfc

! RUN: %kitfc -### -ftapir=opencilk -O2 %s 2>&1 | FileCheck %s

! CHECK: -fc1
! CHECK-SAME: -ftapir=opencilk
! CHECK-SAME: -opencilk-abi-bitcode
! CHECK-SAME: -fstripmine

! It is a pain to check for the actual linker executable. There are far too
! many options depending on the platform, so just check the next line for the
! expected linker flags.
! CHECK-NEXT: -lopencilk
! CHECK-SAME: -lkitrt

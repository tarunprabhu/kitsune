! REQUIRES: kitfc
! REQUIRES: kitsune-opencilk
!
! RUN: %kitfc -### -ftapir=opencilk -O2 -flto %s 2>&1 \
! RUN:     | FileCheck %s -check-prefixes=ALL
!
! RUN: %kitfc -### -ftapir=opencilk -O2 -flto %s \
! RUN:     --tapir-verbose 2>&1 \
! RUN:     | FileCheck %s -check-prefixes=ALL,TAPIR-VERBOSE
!
! RUN: %kitfc -### -ftapir=opencilk -O2 -flto %s \
! RUN:     --kitrt-verbose 2>&1 \
! RUN:     | FileCheck %s -check-prefixes=ALL,KITRT-VERBOSE

! ALL: /ld{{(64)?}}.lld"
! KITRT-VERBOSE: --kitrt-verbose
! TAPIR-VERBOSE: --tapir-verbose
! ALL-SAME: --tapir=opencilk
! ALL-SAME: --tapir-opencilk-abi-bc={{.+}}/libopencilk-abi.bc

end program

! REQUIRES: kitsune-kitfc
!
! This has not beeen implemented yet.
! XFAIL: *
!
! Check that the frontend options make it to the tapir target
!
! RUN: %kitfc --tapir=serial --tapir-verbose \
! RUN:     -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
! RUN:     | FileCheck %s -check-prefixes ALL,COMPILE
!
! RUN: %kitfc --tapir=serial --tapir-verbose --kitrt-verbose \
! RUN:     -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
! RUN:     | FileCheck %s -check-prefixes ALL,RUNTIME
!
! ALL: 'serial' tapir target options
! COMPILE:   Runtime verbose: true
! RUNTIME:   Runtime verbose: true

! FIXME: Need a DO CONCURRENT loop here to ensure that SerialABI is entered

! REQUIRES: kitfc
!
! This has not been implemented yet.
! XFAIL: *
!
! Check that the frontend options make it to the tapir target
!
! RUN: %kitfc --tapir=cuda --tapir-verbose         \
! RUN:     -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
! RUN:     | FileCheck %s -check-prefixes ALL,COMPILE
!
! RUN: %kitfc --tapir=cuda --tapir-verbose --kitrt-verbose \
! RUN:     -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
! RUN:     | FileCheck %s -check-prefixes ALL,RUNTIME
!
! RUN: %kitfc --tapir=cuda --tapir-verbose --tapir-cuda-arch=sm_72 \
! RUN:     -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
! RUN:     | FileCheck %s -check-prefixes ALL,ARCH
!
! RUN: %kitfc --tapir=cuda --tapir-verbose --tapir-threads-per-block=64 %s \
! RUN:     -O2 -S -emit-llvm -o - 2>&1 \
! RUN:     | FileCheck %s -check-prefix TPB
!
! RUN: %kitfc --tapir=cuda --tapir-verbose --tapir-max-threads-per-block=64 %s \
! RUN:     -O2 -S -emit-llvm -o - 2>&1 \
! RUN:     | FileCheck %s -check-prefix MTPB
!
! ALL: 'cuda' tapir target options
! COMPILE:   Runtime verbose: true
! RUNTIME:   Runtime verbose: true
! ARCH:      GPU arch: sm_72
! TPB:       Fixed threads/block: 64
! MTPB:      Max threads/block: 64

! FIXME: We need a DO CONCURRENT loop so CudaABI is entered.

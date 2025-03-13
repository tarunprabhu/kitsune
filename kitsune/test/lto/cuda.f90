! REQUIRES: kitfc
! REQUIRES: kitsune-cuda
!
! RUN: %kitfc -### -ftapir=cuda -O2 -flto %s 2>&1 \
! RUN:     | FileCheck %s -check-prefixes=ALL
!
! RUN: %kitfc -### -ftapir=cuda -O2 -flto %s \
! RUN:     --tapir-verbose 2>&1 \
! RUN:     | FileCheck %s -check-prefixes=ALL,TAPIR-VERBOSE
!
! RUN: %kitfc -### -ftapir=cuda -O2 -flto %s \
! RUN:     --kitrt-verbose 2>&1 \
! RUN:     | FileCheck %s -check-prefixes=ALL,KITRT-VERBOSE
!
! RUN: %kitfc -### -ftapir=cuda -O2 -flto %s \
! RUN:     --tapir-cuda-arch=sm_72 2>&1 \
! RUN:     | FileCheck %s -check-prefixes=ALL,CUDA_ARCH
!
! RUN: %kitfc -### -ftapir=cuda -O2 -flto %s \
! RUN:     --tapir-threads-per-block=64 2>&1 \
! RUN:     | FileCheck %s -check-prefixes=ALL,TPB
!
! RUN: %kitfc -### -ftapir=cuda -O2 -flto %s \
! RUN:     --tapir-max-threads-per-block=128 2>&1 \
! RUN:     | FileCheck %s -check-prefixes=ALL,MTPB

! ALL: /ld{{(64)?}}.lld"
! TAPIR-VERBOSE: --tapir-verbose
! KITRT-VERBOSE: --kitrt-verbose
! ALL-SAME: --tapir=cuda
! CUDA_ARCH: --tapir-cuda-arch=sm_72
! TPB: --tapir-threads-per-block=64
! MTPB: --tapir-max-threads-per-block=128

end program

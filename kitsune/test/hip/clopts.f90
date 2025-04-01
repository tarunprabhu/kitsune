! REQUIRES: kitsune-kitfc
!
! This has not been implemented yet.
! XFAIL: *
!
! Check that the frontend options make it to the tapir target
!
! RUN: %kitfc --tapir=hip --tapir-verbose          \
! RUN:     -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
! RUN:     | FileCheck %s -check-prefixes ALL,COMPILE
!
! RUN: %kitfc --tapir=hip --tapir-verbose --kitrt-verbose \
! RUN:     -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
! RUN:     | FileCheck %s -check-prefixes ALL,RUNTIME
!
! RUN: %kitfc --tapir=hip --tapir-verbose --tapir-hip-arch=gfx906 \
! RUN:     -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
! RUN:     | FileCheck %s -check-prefixes ALL,ARCH
!
! RUN: %kitfc --tapir=hip --tapir-verbose --tapir-threads-per-block=64 %s \
! RUN:     -O2 -S -emit-llvm -o - 2>&1 \
! RUN:     | FileCheck %s -check-prefix TPB
!
! RUN: %kitfc --tapir=hip --tapir-verbose --tapir-max-threads-per-block=64 %s \
! RUN:     -O2 -S -emit-llvm -o - 2>&1 \
! RUN:     | FileCheck %s -check-prefix MTPB
!
! RUN: %kitfc --tapir=hip --tapir-verbose --tapir-hip-sramecc=off %s \
! RUN:     -O2 -S -emit-llvm -o - 2>&1 \
! RUN:     | FileCheck %s -check-prefix SRAMECC_OFF
!
! RUN: %kitfc --tapir=hip --tapir-verbose --tapir-hip-sramecc=on %s \
! RUN:     -O2 -S -emit-llvm -o - 2>&1 \
! RUN:     | FileCheck %s -check-prefix SRAMECC_ON
!
! RUN: %kitfc --tapir=hip --tapir-verbose --tapir-hip-sramecc=any %s \
! RUN:     -O2 -S -emit-llvm -o - 2>&1 \
! RUN:     | FileCheck %s -check-prefix SRAMECC_ANY
!
! RUN: %kitfc --tapir=hip --tapir-verbose --tapir-hip-xnack=off %s \
! RUN:     -O2 -S -emit-llvm -o - 2>&1 \
! RUN:     | FileCheck %s -check-prefix XNACK_OFF
!
! RUN: %kitfc --tapir=hip --tapir-verbose --tapir-hip-xnack=on %s \
! RUN:     -O2 -S -emit-llvm -o - 2>&1 \
! RUN:     | FileCheck %s -check-prefix XNACK_ON
!
! RUN: %kitfc --tapir=hip --tapir-verbose --tapir-hip-xnack=any %s \
! RUN:     -O2 -S -emit-llvm -o - 2>&1 \
! RUN:     | FileCheck %s -check-prefix XNACK_ANY
!
! ALL: 'hip'   tapir target options
! COMPILE:     Runtime verbose: true
! RUNTIME:     Runtime verbose: true
! ARCH:        GPU arch: gfx906
! TPB:         Fixed threads/block: 64
! MTPB:        Max threads/block: 64
! SRAMECC_OFF: SramECC: off
! SRAMECC_ON:  SramECC: on
! SRAMECC_ANY: SramECC: any
! XNACK_OFF:   Xnack: off
! XNACK_ON:    Xnack: on
! XNACK_ANY:   Xnack: any


! REQUIRES: kitfc
!
! -----------------------------------------------------------------------------
!
! RUN: not %flang -### --tapir-cuda-arch=sm_72 --tapir=cuda %s 2>&1    \
! RUN:     | FileCheck %s -check-prefix FRONTEND
!
! FRONTEND: option '--tapir-cuda-arch=' must be used with a Kitsune frontend
!
! -----------------------------------------------------------------------------
!
! The --tapir-cuda-arch is not used if the tapir target is not cuda, or if the
! tapir target is not set.
!
! RUN: %kitfc -### --tapir-cuda-arch=sm_72 %s 2>&1  \
! RUN:     | FileCheck %s -check-prefix UNUSED
! RUN: %kitfc -### --tapir-cuda-arch=sm_72 -ftapir=serial %s 2>&1  \
! RUN:     | FileCheck %s -check-prefix UNUSED
!
! UNUSED-NOT: '--tapir-cuda-arch=sm_72'
!
! -----------------------------------------------------------------------------
!
! RUN: not %kitfc -### --tapir=cuda --tapir-cuda-arch=gfx90a %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix INVALID
!
! INVALID: error: unsupported NVIDIA GPU architecture 'gfx90a'
!
! -----------------------------------------------------------------------------

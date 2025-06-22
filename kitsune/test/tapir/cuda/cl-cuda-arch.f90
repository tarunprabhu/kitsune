! REQUIRES: kitfc
!
! Check that the --tapir-cuda-arch option is handled correctly.
!
! -----------------------------------------------------------------------------
!
! RUN: not %kitfc -### --tapir=cuda --tapir-cuda-arch=gfx906 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix INVALID
!
! INVALID: error: unsupported NVIDIA GPU architecture 'gfx906'
!
! -----------------------------------------------------------------------------
!
! RUN: %kitfc -### --tapir=cuda --tapir-cuda-arch=sm_72 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix OK
!
! OK: -fc1
! OK-SAME: --tapir-cuda-arch=sm_72

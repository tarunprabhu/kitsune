! REQUIRES: kitfc
!
! -----------------------------------------------------------------------------
!
! RUN: not %kitfc -### --tapir=hip --tapir-hip-arch=sm_80 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix INVALID
!
! INVALID: error: unsupported AMD GPU architecture 'sm_80'
!
! -----------------------------------------------------------------------------
!
! RUN: %kitfc -### --tapir=hip --tapir-hip-arch=gfx906 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix OK
!
! OK: -fc1
! OK-SAME: --tapir-hip-arch=gfx906

end program

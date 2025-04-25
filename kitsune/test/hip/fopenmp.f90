! REQUIRES: kitfc
!
! -----------------------------------------------------------------------------
! OpenMP offload cannot be used together with a Kitsune tapir target.
!
! RUN: not %kitfc -### -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN:     -offload-gpu-arch=gfx90a --tapir=hip -nogpulib -c -O2 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix=TAPIR
!
! TAPIR: cannot use OpenMP offload with a tapir target
!
! ----------------------------------------------------------------------------
! Running the kitsune frontend without --tapir is allowed.
!
! RUN: %kitfc -### -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN:     -offload-gpu-arch=gfx90a -nogpulib -c -O2 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix=NOTAPIR
!
! NOTAPIR-NOT: cannot use OpenMP offload with a tapir target

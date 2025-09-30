! REQUIRES: kitfc
!
! ------------------------------------------------------------------------------
! OpenMP offload cannot be used together with a Kitsune tapir target.
!
! RUN: not %kitfc -### -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
! RUN:     --offload-arch=sm_80 -nogpulib -c -O2 %s \
! RUN:     --tapir=cuda --tapir-cuda-arch=sm_80 2>&1 \
! RUN:     | FileCheck %s -check-prefix=TAPIR
!
! TAPIR: cannot use offload with a tapir target
!
! ------------------------------------------------------------------------------
! Running the kitsune frontend without --tapir is ok.
!
! RUN: %kitfc -### -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
! RUN:     --offload-arch=sm_80 -nogpulib -c -O2 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix=NOTAPIR
!
! NOTAPIR-NOT: cannot use offload with a tapir target

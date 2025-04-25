! REQUIRES: kitfc
!
! ------------------------------------------------------------------------------
! OpenMP offload cannot be used together with a Kitsune tapir target.
!
! RUN: not %kitxx -### -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
! RUN:     --cuda-gpu-arch=sm_80 -nocudalib -c -O2 %s \
! RUN:     --tapir=cuda --tapir-cuda-arch=sm_80 \
! RUN:     --libomptarget-nvptx-bc-path=%S/input/nvptx.bc 2>&1 \
! RUN:     | FileCheck %s -check-prefix=TAPIR
!
! TAPIR: cannot use OpenMP offload with a tapir target
!
! ------------------------------------------------------------------------------
! Running the kitsune frontend without --tapir is ok
!
! RUN: %kitxx -### -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
! RUN:     --cuda-gpu-arch=sm_80 -nocudalib -c -O2 %s \
! RUN:     --libomptarget-nvptx-bc-path=%S/input/nvptx.bc 2>&1 \
! RUN:     | FileCheck %s -check-prefix=NOTAPIR
!
! NOTAPIR-NOT: cannot use OpenMP offload with a tapir target

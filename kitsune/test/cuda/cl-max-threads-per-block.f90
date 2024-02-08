! REQUIRES: kitfc
!
! Check that the --tapir-gpu-max-tpb option is handled correctly.
!
! RUN: not %kitfc -### --tapir=cuda --tapir-gpu-max-tpb= %s \
! RUN:     --tapir-cuda-arch=sm_72 2>&1 \
! RUN:     | FileCheck %s -check-prefix MISSING
!
! RUN: not %kitfc -### --tapir=cuda --tapir-gpu-max-tpb=-1 %s \
! RUN:     --tapir-cuda-arch=sm_72 2>&1 \
! RUN:     | FileCheck %s -check-prefix UNDERFLOW
!
! RUN: not %kitfc -### --tapir=cuda --tapir-gpu-max-tpb=0 %s \
! RUN:     --tapir-cuda-arch=sm_72 2>&1 \
! RUN:     | FileCheck %s -check-prefix UNDERFLOW
!
! RUN: %kitfc -### --tapir=cuda --tapir-gpu-max-tpb=1 %s \
! RUN:     --tapir-cuda-arch=sm_72 2>&1 \
! RUN:     | FileCheck %s -check-prefix OK
!
! RUN: %kitfc -### --tapir=cuda --tapir-gpu-max-tpb=1024 %s \
! RUN:     --tapir-cuda-arch=sm_72 2>&1 \
! RUN:     | FileCheck %s -check-prefix OK
!
! RUN: %kitfc -### --tapir=cuda --tapir-gpu-max-tpb=1025 %s \
! RUN:     --tapir-cuda-arch=sm_72 2>&1 \
! RUN:     | FileCheck %s -check-prefix OK
!
! MISSING: error: argument to '{{.+}}' is missing
! UNDERFLOW: error: value of '{{.+}}' must be at least 1
! OK: --tapir-gpu-max-tpb={{[0-9]+}}

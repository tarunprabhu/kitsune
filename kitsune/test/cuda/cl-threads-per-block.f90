! REQUIRES: kitfc
!
! Check that the --tapir-gpu-tpb option is handled correctly.
!
! RUN: not %kitfc -### --tapir=hip --tapir-gpu-tpb= %s \
! RUN:     --tapir-cuda-arch=sm_72 2>&1 \
! RUN:     | FileCheck %s -check-prefix MISSING
!
! RUN: not %kitfc -### --tapir=hip --tapir-gpu-tpb=-1 %s \
! RUN:     --tapir-cuda-arch=sm_72 2>&1 \
! RUN:     | FileCheck %s -check-prefix RANGE
!
! RUN: not %kitfc -### --tapir=hip --tapir-gpu-tpb=0 %s \
! RUN:     --tapir-cuda-arch=sm_72 2>&1 \
! RUN:     | FileCheck %s -check-prefix RANGE
!
! RUN: not %kitfc -### --tapir=hip --tapir-gpu-tpb=1025 %s \
! RUN:     --tapir-cuda-arch=sm_72 2>&1 \
! RUN:     | FileCheck %s -check-prefix RANGE
!
! RUN: %kitfc -### --tapir=hip --tapir-gpu-tpb=1 %s \
! RUN:     --tapir-cuda-arch=sm_72 2>&1 \
! RUN:     | FileCheck %s -check-prefix OK
!
! RUN: %kitfc -### --tapir=hip --tapir-gpu-tpb=1024 %s \
! RUN:     --tapir-cuda-arch=sm_72 2>&1 \
! RUN:     | FileCheck %s -check-prefix OK
!
! MISSING: error: argument to '{{.+}}' is missing
! RANGE: error: value of '{{.+}}' not in range
! OK: --tapir-gpu-tpb={{[0-9]+}}

! REQUIRES: kitfc
!
! Check that the --tapir-max-threads-per-block option is handled correctly.
!
! RUN: not %kitfc -### --tapir=cuda --tapir-max-threads-per-block= %s \
! RUN:     --tapir-cuda-arch=sm_72 2>&1 \
! RUN:     | FileCheck %s -check-prefix MISSING
!
! RUN: not %kitfc -### --tapir=cuda --tapir-max-threads-per-block=-1 %s \
! RUN:     --tapir-cuda-arch=sm_72 2>&1 \
! RUN:     | FileCheck %s -check-prefix UNDERFLOW
!
! RUN: not %kitfc -### --tapir=cuda --tapir-max-threads-per-block=0 %s \
! RUN:     --tapir-cuda-arch=sm_72 2>&1 \
! RUN:     | FileCheck %s -check-prefix UNDERFLOW
!
! RUN: %kitfc -### --tapir=cuda --tapir-max-threads-per-block=1 %s \
! RUN:     --tapir-cuda-arch=sm_72 2>&1 \
! RUN:     | FileCheck %s -check-prefix OK
!
! RUN: %kitfc -### --tapir=cuda --tapir-max-threads-per-block=1024 %s \
! RUN:     --tapir-cuda-arch=sm_72 2>&1 \
! RUN:     | FileCheck %s -check-prefix OK
!
! RUN: %kitfc -### --tapir=cuda --tapir-max-threads-per-block=1025 %s \
! RUN:     --tapir-cuda-arch=sm_72 2>&1 \
! RUN:     | FileCheck %s -check-prefix OK
!
! MISSING: error: argument to '{{.+}}' is missing
! UNDERFLOW: error: value of '{{.+}}' must be at least 1
! OK: --tapir-max-threads-per-block={{[0-9]+}}

! REQUIRES: kitfc
!
! Check that invalid values passed to the --tapir-threads-per-block option
! emit an appropriate error.
!
! RUN: not %kitfc -### --tapir=hip --tapir-threads-per-block= %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix MISSING
!
! RUN: not %kitfc -### --tapir=hip --tapir-threads-per-block=-1 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix RANGE
!
! RUN: not %kitfc -### --tapir=hip --tapir-threads-per-block=0 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix RANGE
!
! RUN: not %kitfc -### --tapir=hip --tapir-threads-per-block=1025 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix RANGE
!
! RUN: %kitfc -### --tapir=hip --tapir-threads-per-block=1 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix TPBOK
!
! RUN: %kitfc -### --tapir=hip --tapir-threads-per-block=1024 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix TPBOK
!
! RUN: not %kitfc -### --tapir=hip --tapir-max-threads-per-block= %s 2>&1  \
! RUN:     | FileCheck %s -check-prefix MISSING
!
! RUN: not %kitfc -### --tapir=hip --tapir-max-threads-per-block=-1 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix UNDERFLOW
!
! RUN: not %kitfc -### --tapir=hip --tapir-max-threads-per-block=0 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix UNDERFLOW
!
! RUN: %kitfc -### --tapir=hip --tapir-max-threads-per-block=1 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix MTPBOK
!
! RUN: %kitfc -### --tapir=hip --tapir-max-threads-per-block=1024 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix MTPBOK
!
! RUN: %kitfc -### --tapir=hip --tapir-max-threads-per-block=1025 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix MTPBOK
!
! MISSING: error: argument to '{{.+}}' is missing
! RANGE: error: value of '{{.+}}' not in range
! UNDERFLOW: error: value of '{{.+}}' must be at least 1
! TPBOK: --tapir-threads-per-block={{[0-9]+}}
! MTPBOK: --tapir-max-threads-per-block={{[0-9]+}}

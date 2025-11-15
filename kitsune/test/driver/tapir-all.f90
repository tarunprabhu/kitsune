! REQUIRES: kitfc
!
! This should test all the tapir targets that may be provided on the command
! line. It should be updated when a new tapir target is added, but there is
! currently no way to enforce this. We do not check for -ftapir here because
! that option is officially deprecated and may be removed at some point.
!
! ------------------------------------------------------------------------------
! The test below should return a success code. We only need to check for that
!
! RUN: %kitfc -### --tapir=nolo -O1 %s
! RUN: %kitfc -### --tapir=serial -O1 %s
! RUN: %if kitsune-cuda %{ \
! RUN:     %kitfc -### --tapir=cuda --tapir-cuda-arch=sm_80 -O1 %s \
! RUN: %}
! RUN: %if kitsune-hip %{ %kitfc -### --tapir=hip -O1 %s %}
! RUN: %if kitsune-opencilk %{ %kitfc -### --tapir=opencilk -O1 %s %}
! RUN: %kitfc -### --tapir=pthreads -O1 %s
!
! ------------------------------------------------------------------------------
! Unknown tapir targets provided to --tapir= should return an error.
!
! RUN: not %kitfc -### --tapir=loremipsum -O1 %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=ERROR
!
! RUN: not %kitfc -### --tapir= -O1 %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=ERROR
!
! ERROR: invalid value '{{.*}}' in '--tapir={{.*}}'
!
! ------------------------------------------------------------------------------
! The tapir targets below have implementations and some measure of support in
! the code. But they have not been maintained and may ave bit-rotted, and are,
! therefore, disabled with limited support even in the build system. If any
! are ever resurrected, they should be moved to the first set of known tapir
! targets.
!
! RUN: not %kitfc -### --tapir=gpuabi -O1 %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=ERROR
! RUN: not %kitfc -### --tapir=lambda -O1 %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=NOT-ENABLED
! RUN: not %kitfc -### --tapir=omptask -O1 %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=NOT-ENABLED
! RUN: not %kitfc -### --tapir=qthreads -O1 %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=NOT-ENABLED
! RUN: not %kitfc -### --tapir=realm -O1 %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=NOT-ENABLED
!
! NOT-ENABLED: tapir target '{{.+}}' was not enabled
!
! ------------------------------------------------------------------------------
! Unlike the tapir targets in the list above, these are likely to be removed
! completely and not resurrected. If that happens, they should be removed from
! here.
!
! RUN: not %kitfc -### --tapir=openmp -O1 %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=NOT-ENABLED

! REQUIRES: kitfc
!
! Check that the --tapir-hip-xnack option is handled correctly.
!
! -----------------------------------------------------------------------------
! RUN: %kitfc -### --tapir=hip --tapir-hip-xnack=on -O1 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefixes ALL,ON
!
! RUN: %kitfc -### --tapir=hip --tapir-hip-xnack=off -O1 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefixes ALL,OFF
!
! RUN: %kitfc -### --tapir=hip --tapir-hip-xnack=any -O1 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefixes ALL,ANY
!
! ALL: -fc1
! ON: --tapir-hip-xnack=on
! OFF: --tapir-hip-xnack=off
! ANY: --tapir-hip-xnack=any
!
! -----------------------------------------------------------------------------
!
! RUN: not %kitfc -### --tapir=hip --tapir-hip-xnack= -O1 %s 2>&1  \
! RUN:     | FileCheck %s -check-prefix MISSING
!
! MISSING: error: argument to '--tapir-hip-xnack=' is missing
!
! -----------------------------------------------------------------------------
!
! RUN: not %kitfc -### --tapir=hip --tapir-hip-xnack=ignore -O1 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix IGNORE
!
! IGNORE: error: invalid argument 'ignore' to -tapir-hip-xnack=
!
! -----------------------------------------------------------------------------

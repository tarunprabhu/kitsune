! REQUIRES: kitfc
!
! Check that the --tapir-hip-sramecc option is handled correctly.
!
! -----------------------------------------------------------------------------
! RUN: %kitfc -### --tapir=hip --tapir-hip-sramecc=on %s 2>&1 \
! RUN:     | FileCheck %s -check-prefixes ALL,ON
!
! RUN: %kitfc -### --tapir=hip --tapir-hip-sramecc=off %s 2>&1 \
! RUN:     | FileCheck %s -check-prefixes ALL,OFF
!
! RUN: %kitfc -### --tapir=hip --tapir-hip-sramecc=any %s 2>&1 \
! RUN:     | FileCheck %s -check-prefixes ALL,ANY
!
! ALL: -fc1
! ON: --tapir-hip-sramecc=on
! OFF: --tapir-hip-sramecc=off
! ANY: --tapir-hip-sramecc=any
!
! -----------------------------------------------------------------------------
!
! RUN: not %kitfc -### --tapir=hip --tapir-hip-sramecc= %s 2>&1  \
! RUN:     | FileCheck %s -check-prefix MISSING
!
! MISSING: error: argument to '--tapir-hip-sramecc=' is missing
!
! -----------------------------------------------------------------------------
!
! RUN: not %kitfc -### --tapir=hip --tapir-hip-sramecc=ignore %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix IGNORE
!
! IGNORE: error: invalid argument 'ignore' to -tapir-hip-sramecc=
!
! -----------------------------------------------------------------------------

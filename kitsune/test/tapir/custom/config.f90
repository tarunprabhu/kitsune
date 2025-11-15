! REQUIRES: kitfc
!
! The 'custom' tapir target does not use a configuration file.
!
! RUN: %kitfc -### --tapir=custom --tapir-plugin=plugin.ext -O1 %s 2>&1 \
! RUN:     | FileCheck %s
!
! CHECK-NOT: Configuration file:

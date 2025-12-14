! REQUIRES: kitsune-examples, kitfc
!
! The 'custom' tapir target does not use a configuration file. Even if a file
! named custom.cfg is present, it is ignored. The name custom.cfg is used here
! since the names of the configuration files of the other tapir targets are
! of the form `<tapir-target>.cfg`.
!
! RUN: %kitfc -### --tapir=custom --tapir-plugin=%kit-tt-plugin-demo -O1 \
! RUN:     %s 2>&1 \
! RUN:     | FileCheck %s
!
! RUN: %kitfc -### --tapir=custom --tapir-plugin=%kit-tt-plugin-demo -O1 \
! RUN:     --config-kitsune-dir=%S/input %s 2>&1 \
! RUN:     | FileCheck %s
!
! RUN: %kitfc -### --tapir=custom --tapir-plugin=%kit-tt-plugin-demo -O1 \
! RUN:     --config-system-dir=%S/input %s 2>&1 \
! RUN:     | FileCheck %s
!
! RUN: %kitfc -### --tapir=custom --tapir-plugin=%kit-tt-plugin-demo -O1 \
! RUN:     --config-user-dir=%S/input %s 2>&1 \
! RUN:     | FileCheck %s
!
! CHECK-NOT: Configuration file: {{.*}}/input/custom.cfg

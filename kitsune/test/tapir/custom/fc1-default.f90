! REQUIRES: kitfc
!
! -----------------------------------------------------------------------------
! Check that the default options added to the internal command lines (for -fc1)
! are as expected.
!
! RUN: %kitfc -### --tapir=custom --tapir-plugin=plugin.ext -O2 %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=FC1
!
! FC1: -fc1
! FC1-SAME: --tapir=custom
! FC1-SAME: --tapir-plugin=
!
! For the 'custom' tapir target, stripmining is disabled by default.
!
! FC1-NOT: -fstripmine
!
! -----------------------------------------------------------------------------
! Check that the stripmine pass is disabled by default. The test below is
! conditional on the examples being built because a valid plugin must be passed
! to the command below.
!
! RUN: %if kitsune-examples %{ \
! RUN:     %kitfc -O2 -S -emit-llvm -o /dev/null %s \
! RUN:         -mllvm -print-pipeline-passes \
! RUN:         --tapir=custom --tapir-plugin=%kit-tt-plugin-demo \
! RUN:         | FileCheck %s -check-prefix STRIPMINE-PASS \
! RUN: %}
!
! STRIPMINE-PASS-NOT: loop-stripmine
!
! ------------------------------------------------------------------------------

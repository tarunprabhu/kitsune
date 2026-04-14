! REQUIRES: kitfc
!
! ------------------------------------------------------------------------------
! Check that the default options added to the internal command lines (for -fc1
! and the linker) are as expected.
!
! RUN: %kitfc -### --tapir=nolo -O2 %s 2>&1 | FileCheck %s
!
! CHECK: -fc1
! CHECK-SAME: --tapir=nolo
! CHECK-SAME: -fstripmine
!
! This pass adds nothing to the linker, so there is nothing to check for. But
! we do check for the absence of libkitrt.
!
! CHECK-NOT: -lkitrt
!
! ------------------------------------------------------------------------------
! Check that the stripmine pass is not enabled by default.
!
! RUN: %kitfc -mllvm -print-pipeline-passes -O2 --tapir=nolo \
! RUN:     -S -emit-llvm %s | FileCheck %s -check-prefix STRIPMINE-PASS
!
! STRIPMINE-PASS: loop-stripmine
!
! ------------------------------------------------------------------------------


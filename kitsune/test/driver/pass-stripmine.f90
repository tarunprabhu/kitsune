! REQUIRES: kitfc
!
! Check that the stripmine pass is enabled/disabled as expected when the
! -fstripmine and -fno-stripmine options are used. This checks that the pipeline
! tuning object is setup correctly. This requires a tapir target, so we just use
! 'serial' since it is guaranteed to be available.
!
! TODO: It may be better to check this for all the tapir targets that are
! enabled in case the code that this exercises changes in a
! tapir-target-dependent way.
!
! RUN: %kitfc -mllvm -print-pipeline-passes -O2 -fstripmine --tapir=serial \
! RUN:     -S -emit-llvm -o /dev/null %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix STRIPMINE-PASS
!
! RUN: %kitfc -mllvm -print-pipeline-passes -O2 -fno-stripmine --tapir=serial \
! RUN:     -S -emit-llvm -o /dev/null %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix NO-STRIPMINE-PASS
!
! STRIPMINE-PASS: loop-stripmine
! NO-STRIPMINE-PASS-NOT: loop-stripmine

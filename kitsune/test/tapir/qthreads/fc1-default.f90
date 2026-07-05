! REQUIRES: kitfc
!
! ------------------------------------------------------------------------------
! Check that the default options added to the internal command lines (for -fc1
! and the linker) are as expected.
!
! RUN: %kitfc -### --tapir=qthreads -O2 %s 2>&1 | FileCheck %s
!
! CHECK: -fc1
! CHECK-SAME: --tapir=qthreads
!
! CHECK-NOT: -fstripmine
!
! We check for the absence of certain libraries that used to be linked
! explicitly in the past, but are not any longer. Calls to functions provided
! by these libraries should not be added directly by any lowering passes.
! Instead, a wrapper should be provided in libkitrt, and that should be called.
!
! CHECK-NOT: -lqthread
!
! The next line is expected to be the linker invocation. Since it is difficult
! to reliably check the name of the linker executable, just check for the
! expected linker flags.
!
! CHECK-NEXT: -lkitrt
!
! ------------------------------------------------------------------------------
! Check that the stripmine pass is disabled by default. This checks that the
! pipeline tuning options object is setup correctly.
!
! RUN: %kitfc -mllvm -print-pipeline-passes -O2 --tapir=qthreads \
! RUN:     -S -emit-llvm -o /dev/null %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix STRIPMINE-PASS
!
! STRIPMINE-PASS-NOT: loop-stripmine
!
! ------------------------------------------------------------------------------

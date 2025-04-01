! REQUIRES: kitfc
!
! Check that the default options added to the internal command lines (for -fc1
! and the linker) are as expected.
!
! -print-pipeline-passes currently does not work with flang, but has been
! implemented upstream. This should pass once we merge with upstream.
! XFAIL: *
!
! ------------------------------------------------------------------------------
! RUN: %kitfc -### -ftapir=hip -O2 %s 2>&1 | FileCheck %s
! RUN: %kitfc -### --tapir=hip -O2 %s 2>&1 | FileCheck %s
!
! CHECK: -fc1
! CHECK-SAME: --tapir=hip
!
! Strip-mining is disabled by default on GPU tapir targets.
! CHECK-NOT: -fstripmine
!
! It is a pain to check for the actual linker executable. There are far too
! many options depending on the platform, so just check the next line for the
! expected linker flags.
!
! CHECK-NEXT: -lkitrt
! CHECK-SAME: -lamdhip64
!
! ------------------------------------------------------------------------------
! Check that the stripmine pass is disabled by default. This checks that the
! the pipeline tuning options object value is set correctly by default.
!
! RUN: %kitfc -mllvm -print-pipeline-passes -O2 -fstripmine -ftapir=hip \
! RUN:      -S -emit-llvm %s | FileCheck %s -check-prefix STRIPMINE-PASS
!
! STRIPMINE-PASS: loop-stripmine

end program

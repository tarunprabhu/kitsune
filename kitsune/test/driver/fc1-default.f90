! REQUIRES: kitfc
!
! Check that the default options added to the internal command lines (for -cc1
! and the linker) are as expected. There are corresponding tests that are
! tapir-target specific. Those generally check that the external libraries
! needed by that specific tapir target are linked. This is intended to check
! that libraries that are required by the "non-tapir-target-specific" parts of
! Kitsune's runtime are linked correctly.
!
! While this test is intended to be "independent" of a specific tapir targets,
! we cannot actually test it without using a tapir target. We pick 'serial'
! because:
!
!   1. It is guaranteed to be available
!   2. It does not require any external libraries, but does use libkitrt, so
!      it is reasonable to check for everything here.
!
! We could have added this to the test in transforms/tapir/serial, but that
! would make the intent of the test less clear.
!
!
! RUN: %kitfc -### --tapir=serial -O2 %s 2>&1 | FileCheck %s
!
!
! CHECK: -fc1
! CHECK-SAME: --tapir=serial
!
! CHECK-NOT: -fstripmine
!
! The next line is expected to be the linker invocation. Since it is difficult
! to reliably check the name of the linker executable, just check for the
! expected linker flags.
!
! CHECK-NEXT: "-lkitrt"
! CHECK-SAME: "-ldl"
! CHECK-SAME: "-lm"
! CHECK-SAME: "-lpthread"
! CHECK-SAME: "-lrt"

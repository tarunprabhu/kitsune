! REQUIRES: kitfc

! -print-pipeline-passes currently does not work with flang, but has been
! implemented upstream. This should pass once we merge with upstream.
! XFAIL: *

! RUN: %kitfc -### -ftapir=hip -O2 %s 2>&1 | FileCheck %s

! CHECK: -fc1
! CHECK-SAME: -ftapir=hip

! Strip-mining is disabled by default on GPU tapir targets.
! CHECK-NOT: -fstripmine

! It is a pain to check for the actual linker executable. There are far too
! many options depending on the platform, so just check the next line for the
! expected linker flags.
! CHECK-NEXT: -lkitrt
! CHECK-SAME: -lhiprt
! CHECK-SAME: -lhip

! Check that the stripmine pass is enabled/disabled correctly
! RUN: %kitfc -mllvm -print-pipeline-passes -O2 -fstripmine -ftapir=hip \
! RUN:      -S -emit-llvm %s | FileCheck %s -check-prefix STRIPMINE-PASS

! STRIPMINE-PASS: loop-stripmine

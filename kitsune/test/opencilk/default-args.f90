! REQUIRES: kitfc
!
! FIXME: The checks for required arguments have not been enabled in flang yet
! XFAIL: *
!
! Check that the default options added to the internal command lines (for -fc1
! and the linker) are as expected.
!
! ------------------------------------------------------------------------------
! RUN: %kitfc -### -ftapir=opencilk -O2 %s 2>&1 | FileCheck %s
! RUN: %kitfc -### --tapir=opencilk -O2 %s 2>&1 | FileCheck %s
!
! CHECK: -fc1
! CHECK-SAME: --tapir=opencilk
! CHECK-SAME: --tapir-opencilk-runtime-bc
!
! For OpenCilk, stripmining is enabled by default.
!
! CHECK-SAME: -fstripmine
!
! It is a pain to check for the actual linker executable. There are far too
! many options depending on the platform, so just check the next line for the
! expected linker flags.
!
! CHECK-NEXT: -lopencilk
! CHECK-SAME: -lkitrt
!
! ------------------------------------------------------------------------------
! Check that the stripmine pass is enabled by default. This checks that the
! the pipeline tuning options object value is set correctly by default.
!
! RUN: %kitfc -mllvm -print-pipeline-passes -O2 -ftapir=opencilk \
! RUN:      -S -emit-llvm %s | FileCheck %s -check-prefix STRIPMINE-PASS
!
! STRIPMINE-PASS: loop-stripmine
!
! ------------------------------------------------------------------------------
! Check that an error is emitted if any of the required options are not
! provided
!
! RUN: not %kitfc -fc1 --tapir=opencilk %s -o /dev/null 2>&1 \
! RUN:     | FileCheck %s -check-prefix=MISSING_RUNTIME_BC
!
! MISSING_RUNTIME_BC: missing required option '--tapir-opencilk-runtime-bc='
!
! ------------------------------------------------------------------------------

end program

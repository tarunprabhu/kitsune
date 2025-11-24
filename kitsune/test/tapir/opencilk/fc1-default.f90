! REQUIRES: kitfc
!
! ------------------------------------------------------------------------------
! Check that the default options added to the internal command lines (for -fc1
! and the linker) are as expected.
!
! RUN: %if system-darwin %{ \
! RUN:     %kitfc -### --tapir=opencilk -O2 %s %sysroot 2>&1 \
! RUN:         | FileCheck %s --check-prefixes=ALL,DARWIN \
! RUN: %} %else %{ \
! RUN:     %kitfc -### --tapir=opencilk -O2 %s %sysroot 2>&1 \
! RUN:         | FileCheck %s --check-prefixes=ALL,X86 \
! RUN: %}
!
! ALL: -fc1
! ALL-SAME: --tapir=opencilk
! ALL-SAME: --tapir-opencilk-runtime-bc
!
! For opencilk, stripmining is enabled by default.
!
! ALL-SAME: -fstripmine
!
! It is a pain to check for the actual linker executable. There are far too
! many options depending on the platform, so just check the next line for the
! expected linker flags.
!
! DARWIN-NEXT: "-lopencilk-personality-c_osx_dynamic"
! DARWIN-SAME: "-lopencilk_osx_dynamic"
!
! X86-NEXT: "-lopencilk-personality-c"
! X86-SAME: "-lopencilk"
!
! ALL-SAME: "-lkitrt"
!
! ------------------------------------------------------------------------------
! Check that the stripmine pass is enabled by default. This checks that the
! the pipeline tuning options object value is set correctly by default.
!
! RUN: %kitfc -mllvm -print-pipeline-passes -O2 --tapir=opencilk \
! RUN:     -S -emit-llvm %s %sysroot 2>&1 \
! RUN:     | FileCheck %s -check-prefix STRIPMINE-PASS
!
! STRIPMINE-PASS: loop-stripmine
!
! ------------------------------------------------------------------------------

end program

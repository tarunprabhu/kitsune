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
! ALL-SAME: -fstripmine
!
! We check for the absence of certain libraries that used to be linked
! explicitly in the past, but are not any longer. Calls to functions provided
! by these libraries should not be added directly by any lowering passes.
! Instead, a wrapper should be provided in libkitrt, and that should be called.
!
! X86-NOT: "-lopencilk"
! DARWIN-NOT: "-lopencilk_osx_dynamic"
!
! The next line is expected to be the linker invocation. Since it is difficult
! to reliably check the name of the linker executable, just check for the
! expected linker flags.
!
! DARWIN-NEXT: "-lopencilk-personality-c_osx_dynamic"
! DARWIN-SAME: "-lopencilk_osx_dynamic"
!
! X86-NEXT: "-lopencilk-personality-c"
! ALL-SAME: "-lkitrt"
!
! ------------------------------------------------------------------------------
! Check that the stripmine pass is enabled by default. This checks that the
! pipeline tuning options object is setup correctly.
!
! RUN: %kitfc -mllvm -print-pipeline-passes -O2 --tapir=opencilk \
! RUN:     -S -emit-llvm -o /dev/null %s %sysroot 2>&1 \
! RUN:     | FileCheck %s -check-prefix STRIPMINE-PASS
!
! STRIPMINE-PASS: loop-stripmine
!
! ------------------------------------------------------------------------------

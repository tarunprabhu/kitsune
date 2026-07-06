// -----------------------------------------------------------------------------
// Check that the default options added to the internal command lines (for -cc1
// and the linker) are as expected.
//
// RUN: %if system-darwin %{ \
// RUN:     %kitcc -### --tapir=opencilk -O2 %s %sysroot 2>&1 \
// RUN:         | FileCheck %s --check-prefixes=ALL,DARWIN \
// RUN: %} %else %{ \
// RUN:     %kitcc -### --tapir=opencilk -O2 %s %sysroot 2>&1 \
// RUN:         | FileCheck %s --check-prefixes=ALL,X86 \
// RUN: %}
//
// ALL: -cc1
// ALL-SAME: --tapir=opencilk
// ALL-SAME: --tapir-opencilk-runtime-bc
// ALL-SAME: -fstripmine
//
// The next line is expected to be the linker invocation. Since it is difficult
// to reliably check the name of the linker executable, just check for the
// expected linker flags.
//
// DARWIN-NEXT: "-lopencilk_osx_dynamic"
// DARWIN-SAME: "-lopencilk-personality-c_osx_dynamic"
//
// X86-NEXT: "-lopencilk"
// X86-SAME: "-lopencilk-personality-c"
//
// ALL-SAME: "-lkitrt"
// ALL-NOT: "-l{{[^"]*}}c++"
//
// -----------------------------------------------------------------------------
// Check that the stripmine pass is enabled by default. This checks that the
// pipeline tuning options object is setup correctly.
//
// RUN: %kitcc -mllvm -print-pipeline-passes -O2 --tapir=opencilk \
// RUN:     -S -emit-llvm -o /dev/null %s %sysroot 2>&1 \
// RUN:     | FileCheck %s -check-prefix STRIPMINE-PASS
//
// STRIPMINE-PASS: loop-stripmine
//
// -----------------------------------------------------------------------------

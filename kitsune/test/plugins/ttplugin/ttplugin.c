// REQUIRES: kitsune-examples
//
// Check that a tapir target plugin works as expected on C code. We use the
// tapir target plugin demo for consistency with the way LLVM pass plugins are
// tested.
//
// -----------------------------------------------------------------------------
// Check that any compiler and linker options required by the plugin are added
// to the compiler and linker invocations. The linker invocation is assumed to
// be on the line immediately after the invocation to the compiler in the -###
// output below.
//
// RUN: %kitcc -### --tapir=custom --tapir-plugin=%kit-ttplugin-demo %s \
// RUN:     -o /dev/null -O2 2>&1 \
// RUN:     | FileCheck %s --check-prefixes=ARGS
//
// ARGS: -cc1
// ARGS-SAME: "-O"
// ARGS-NEXT: "-L/path/to/something/that/does/not/exist"
//
// -----------------------------------------------------------------------------
// Check that the plugin modified the code in the expected way.
//
// RUN: %kitcc --tapir=custom --tapir-plugin=%kit-ttplugin-demo %s \
// RUN:     -S -emit-llvm -o - -O2 \
// RUN:     | FileCheck %s --check-prefix=BOOKEND
//
// BOOKEND: call void @bookend
// BOOKEND-NEXT: call {{.*}}void @mset{{[^(]+}}(
// BOOKEND-NEXT: call void @bookend

#include <kitsune.h>

void mset(int *ptr, long n) {
  // clang-format: off
  forall(long i = 0; i < n; ++i) { ptr[i] = i; }
  // clang-format: on
}

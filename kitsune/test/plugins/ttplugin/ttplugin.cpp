// REQUIRES: kitsune-examples
//
// Check that a tapir target plugin works as expected on C++ code. We use the
// tapir target plugin demo for consistency with the way LLVM pass plugins are
// tested.
//
// RUN: %kitxx --tapir=custom --tapir-plugin=%kit-ttplugin-demo %s \
// RUN:     -S -emit-llvm -o - -O2 \
// RUN:     | FileCheck %s --check-prefix=BOOKEND
//
// BOOKEND: call void @bookend
// BOOKEND-NEXT: call {{.*}}void @mset{{[^(]+}}(
// BOOKEND-NEXT: call void @bookend

#include <kitsune.h>

extern "C" void mset(int *ptr, long n) {
  // clang-format: off
  forall(long i = 0; i < n; ++i) { ptr[i] = i; }
  // clang-format: on
}

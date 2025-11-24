// Check that setting the tapir::target attribute with the value "serial"
// overrides the primary tapir target.
//
// RUN: %kitxx -O1 --tapir=pthreads -Xclang -disable-llvm-passes \
// RUN:     -S -emit-llvm -o - %s %sysroot \
// RUN:     | FileCheck %s
//
// CHECK: !{!"tapir.loop.target", i32 1}

#include <kitsune.h>

void f(int *a, int scale, size_t n) {
  // clang-format off
  [[tapir::target("serial")]]
  forall (size_t i = 0; i < n; ++i) {
    a[i] *= scale;
  }
  // clang-format on
}

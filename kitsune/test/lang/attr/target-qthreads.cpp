// REQUIRES: kitsune-qthreads
//
// Check that setting the tapir::target attribute with the value "qthreads"
// overrides the primary tapir target.
//
// RUN: %kitxx -O1 --tapir=serial -Xclang -disable-llvm-passes \
// RUN:     -S -emit-llvm -o - %s 2>&1 \
// RUN:     | FileCheck %s
//
// CHECK: !{!"tapir.loop.target", i32 32}

#include <kitsune.h>

void f(int *a, size_t n) {
  // clang-format off
  [[tapir::target("qthreads")]]
  forall (size_t i = 0; i < n; ++i) {
    a[i] = i;
  }
  // clang-format on
}

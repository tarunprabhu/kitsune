// REQUIRES: kitsune-hip
//
// Check that setting the tapir::target attribute with the value "hip"
// overrides the primary tapir target.
//
// RUN: %kitxx -O1 --tapir=serial -Xclang -disable-llvm-passes \
// RUN:     -S -emit-llvm -o - %s \
// RUN:     | FileCheck %s
//
// CHECK-DAG: !{!"tapir.loop.target", i32 4}

#include <kitsune.h>

void f(size_t *a, size_t n) {
  // clang-format off
  [[tapir::target("hip")]]
  forall (size_t i = 0; i < n; ++i) {
    a[i] = i;
  }
  // clang-format on
}

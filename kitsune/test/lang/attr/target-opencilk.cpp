// REQUIRES: kitsune-opencilk
//
// Check that setting the tapir::target attribute with the value "opencilk"
// overrides the primary tapir target.
//
// RUN: %kitxx -O1 --tapir=serial -Xclang -disable-llvm-passes \
// RUN:     -S -emit-llvm -o - %s 2>&1 \
// RUN:     | FileCheck %s
//
// CHECK-DAG: !{!"tapir.loop.target", i32 8}

#include <kitsune.h>

void f(int *a, int scale, size_t n) {
  // clang-format off
  [[tapir::target("opencilk")]]
  forall (size_t i = 0; i < n; ++i) {
    a[i] *= scale;
  }
  // clang-format on
}

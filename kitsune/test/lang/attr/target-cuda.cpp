// REQUIRES: kitsune-cuda
//
// Check that setting the tapir::target attribute with the value "cuda"
// overrides the primary tapir target.
//
// RUN: %kitxx -O1 --tapir=serial -Xclang -disable-llvm-passes \
// RUN:     -S -emit-llvm -o - %s 2>&1 \
// RUN:     | FileCheck %s
//
// CHECK: !{!"tapir.loop.target", i32 2}

#include <kitsune.h>

void f(size_t *a, size_t n) {
  // clang-format off
  [[tapir::target("cuda")]]
  forall (size_t i = 0; i < n; ++i) {
    a[i] = i;
  }
  // clang-format on
}

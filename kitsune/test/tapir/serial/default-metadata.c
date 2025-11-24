// Check that the correct metadata is added to a tapir loop by default.
//
// RUN: %kitcc --tapir=serial -O1 %s %sysroot \
// RUN:     -Xclang -disable-llvm-passes -S -emit-llvm -o - \
// RUN:     | FileCheck %s
//
// CHECK-DAG: !{!"tapir.loop.spawn.strategy", i32 1}
// CHECK-DAG: !{!"tapir.loop.target", i32 1}

#include <kitsune.h>

void f(int *a, int scale, size_t n) {
  // clang-format off
  forall (size_t i = 0; i < n; ++i) {
    a[i] *= scale;
  }
  // clang-format on
}

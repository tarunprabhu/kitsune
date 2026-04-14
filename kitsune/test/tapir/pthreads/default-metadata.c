// Check that the correct metadata is added to a tapir loop by default.
//
// RUN: %kitcc --tapir=pthreads -O1 %s %sysroot \
// RUN:     -Xclang -disable-llvm-passes -S -emit-llvm -o - 2>&1 \
// RUN:     | FileCheck %s
//
// CHECK-DAG: !{!"tapir.loop.spawn.strategy", i32 4}
// CHECK-DAG: !{!"tapir.loop.target", i32 1024}

#include <kitsune.h>

void f(size_t *a, size_t n) {
  // clang-format off
  forall (size_t i = 0; i < n; ++i) {
    a[i] = i;
  }
  // clang-format on
}

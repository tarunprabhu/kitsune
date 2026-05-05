// Check that the correct metadata is added to a tapir loop by default.
//
// RUN: %kitcc --tapir=cuda --tapir-cuda-arch=sm_72 -O1 %s \
// RUN:     -Xclang -disable-llvm-passes -S -emit-llvm -o - \
// RUN:     | FileCheck %s
//
// CHECK-DAG: !{!"tapir.loop.grainsize", i32 0}
// CHECK-DAG: !{!"tapir.loop.spawn.strategy", i32 3}
// CHECK-DAG: !{!"tapir.loop.target", i32 2}
// CHECK-DAG: !{!"tapir.loop.threads.per.block", i32 0}

#include <kitsune.h>

void f(size_t *a, size_t n) {
  // clang-format off
  forall (size_t i = 0; i < n; ++i) {
    a[i] = i;
  }
  // clang-format on
}

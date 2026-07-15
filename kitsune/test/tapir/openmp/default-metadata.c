// Check that the correct metadata is added to a tapir loop by default.
//
// RUN: %kitcc --tapir=openmp -O1 %s %sysroot \
// RUN:     -Xclang -disable-llvm-passes -S -emit-llvm -o - 2>&1 \
// RUN:     | FileCheck %s

#include <kitsune.h>

void f(size_t *a, size_t n) {
  // clang-format off
  forall (size_t i = 0; i < n; ++i) {
    a[i] = i;
  }
  // clang-format on
}

// The check lines are at the end because the loop name contains the line number
// of the forall statement. We want that to remain consistent, even if we change
// the default metadata that is added to a loop.
//
// CHECK-DAG: !{!"tapir.loop.name", !"default-metadata.c:11:3"}
// CHECK-DAG: !{!"tapir.loop.spawn.strategy", i32 4}
// CHECK-DAG: !{!"tapir.loop.target", i32 512}

// Check that the value of threads per block in the kitsune::launch attribute
// is lowered to the correct metadata.
//
// RUN: %kitxx --tapir=cuda --tapir-cuda-arch=sm_72 -O1 %s \
// RUN:     -Xclang -disable-llvm-passes -S -emit-llvm -o - \
// RUN:     | FileCheck %s --check-prefix=ATTR
//
// ATTR: !{!"tapir.loop.threads.per.block", i32 57}

#include <kitsune.h>

void f(int *a, int n) {
  // clang-format off
  [[kitsune::launch(57)]]
  forall (int i = 0; i < n; ++i)
    a[i] = i;
  // clang-format on
}

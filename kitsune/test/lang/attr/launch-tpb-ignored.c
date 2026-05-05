// The kitsune::launch attribute is not lowered to metadata if the tapir target
// is not a GPU-centric tapir target.
//
// RUN: %if kitsune-opencilk %{ \
// RUN:   %kitcc --tapir=opencilk -O1 -S -emit-llvm -o - %s \
// RUN:       -Xclang -disable-llvm-passes \
// RUN:       | FileCheck %s \
// RUN: %}
//
// RUN: %kitcc --tapir=openmp -O1 -S -emit-llvm -o - %s \
// RUN:     -Xclang -disable-llvm-passes \
// RUN:     | FileCheck %s
//
// RUN: %kitcc --tapir=pthreads -O1 -S -emit-llvm -o - %s \
// RUN:     -Xclang -disable-llvm-passes \
// RUN:     | FileCheck %s
//
// RUN: %if kitsune-qthreads %{ \
// RUN:   %kitcc --tapir=qthreads -O1 -S -emit-llvm -o - %s \
// RUN:       -Xclang -disable-llvm-passes \
// RUN:       | FileCheck %s \
// RUN: %}
//
// RUN: %kitcc --tapir=serial -O1 -S -emit-llvm -o - %s \
// RUN:     -Xclang -disable-llvm-passes \
// RUN:     | FileCheck %s
//
// CHECK-NOT: "tapir.loop.threads.per.block"

#include <kitsune.h>

void f(int *a, int n) {
  // clang-format off
  [[kitsune::launch(57)]]
  forall (int i = 0; i < n; ++i)
    a[i] = i;
  // clang-format on
}

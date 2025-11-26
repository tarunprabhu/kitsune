// Check that the correct tapir.loop.target metadata is added to forall loops.
//
// -----------------------------------------------------------------------------
// If the tapir target is nolo, the loop must not have any loop.target metadata.
//
// RUN: %kitxx --tapir=nolo -O1 %sysroot -S -emit-llvm -o - %s \
// RUN:     -Xclang -disable-llvm-passes \
// RUN:     | FileCheck %s -check-prefix NOLO
//
// NOLO-NOT: "tapir.loop.target"
//
// -----------------------------------------------------------------------------
// If the tapir target is not nolo, the tapir loops must have a loop.target
// metadata whose value is the integer representation of the tapir target. We
// check more than one target here just in case. The targets are all guaranteed
// to be built.
//
// RUN: %kitxx --tapir=serial -O1 %sysroot -S -emit-llvm -o - %s \
// RUN:     -Xclang -disable-llvm-passes \
// RUN:     | FileCheck %s -check-prefix SERIAL
//
// SERIAL: !{!"tapir.loop.target", i32 1}
//
// RUN: %kitxx --tapir=pthreads -O1 %sysroot -S -emit-llvm -o - %s \
// RUN:     -Xclang -disable-llvm-passes \
// RUN:     | FileCheck %s -check-prefix PTHREADS
//
// PTHREADS: !{!"tapir.loop.target", i32 1024}
//
// -----------------------------------------------------------------------------

#include <kitsune.h>

extern "C" void f(int *a, int scale, size_t n) {
  // clang-format off
  forall (size_t i = 0; i < n; ++i) {
    a[i] *= scale;
  }
  // clang-format on
}

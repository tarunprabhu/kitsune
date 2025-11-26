// Check that the tapir::target attribute attached to a kokkos::parallel_for
// call is handled correctly when lowering to LLVM-IR. The tapir target
// specified on the command should always be ignored, even if it is 'nolo'. A
// tapir.loop.metadata metadata node must be attached to the loop whose value
// must be the integer representation of the tapir target
//
// RUN: %kitxx -DWITH_ATTR --kokkos --kokkos-no-init --tapir=serial \
// RUN:     %sysroot -O1 -S -emit-llvm -o - %s \
// RUN:     -Xclang -disable-llvm-passes 2>&1 \
// RUN:     | FileCheck %s
//
// RUN: %kitxx -DWITH_ATTR --kokkos --kokkos-no-init --tapir=nolo \
// RUN:     %sysroot -O1 -S -emit-llvm -o - %s \
// RUN:     -Xclang -disable-llvm-passes 2>&1 \
// RUN:     | FileCheck %s
//
// CHECK: !{!"tapir.loop.target", i32 1024}

#include <Kokkos_Core.hpp>

extern "C" void f(int *a, int scale, size_t n) {
  // clang-format off
  [[tapir::target("pthreads")]]
  Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
    a[i] *= scale;
  });
  // clang-format on
}

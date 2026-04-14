// Check that the correct tapir.loop.target metadata is added to the tapir
// loops created by lowering kokkos::parallel_for.
//
// -----------------------------------------------------------------------------
// If the tapir target is nolo, the loop generated from the kokkos::parallel_for
// must not have any loop.target metadata.
//
// RUN: %kitxx --kokkos --kokkos-no-init --tapir=nolo -O1 \
// RUN:     %sysroot -c -emit-llvm -o - %s \
// RUN:     -Xclang -disable-llvm-passes 2>&1 \
// RUN:     | FileCheck %s -check-prefix NOLO
//
// NOLO-NOT: "tapir.loop.target"
//
// -----------------------------------------------------------------------------
// If the tapir target is not nolo, the loop generated from the
// kokkos::parallel_for must have a loop.target metadata whose value is the
// integer representation of the tapir target.
//
// RUN: %kitxx --kokkos --kokkos-no-init --tapir=serial -O1 \
// RUN:     %sysroot -S -emit-llvm -o - %s \
// RUN:     -Xclang -disable-llvm-passes 2>&1 \
// RUN:     | FileCheck %s -check-prefix SERIAL
//
// SERIAL: !{!"tapir.loop.target", i32 1}
//
// -----------------------------------------------------------------------------

#include <Kokkos_Core.hpp>

extern "C" void f(size_t *a, size_t n) {
  // clang-format off
  Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
    a[i] = i;
  });
  // clang-format on
}

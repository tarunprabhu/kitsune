// Check that the correct warning is emitted when an unsupported policy is seen
// in a parallel_for and that we fall back to the default C++ lowering.
//
// In this case, we have used a TeamPolicy, but if we ever do support
// TeamPolicy's, this should be replaced with something else that is not
// supported.
//
// RUN: %kitxx -fkokkos -fkokkos-no-init -O1 --tapir=nolo -S -emit-llvm \
// RUN:     -Xclang -verify -fno-inline -o - %sysroot %s \
// RUN:     | FileCheck %s

#include <Kokkos_Core.hpp>

extern "C" void ext();

// CHECK-LABEL: void @f
// CHECK-SAME: %[[N:[^)]+]])
// CHECK: call {{.+}}void @"_ZN6Kokkos12parallel_for
extern "C" void f(int n) {
  using Team = typename Kokkos::TeamPolicy<>::member_type;

  auto teamPolicy = typename Kokkos::TeamPolicy<>(n, Kokkos::AUTO);
  // clang-format off
  // expected-warning-re@+1 {{kokkos: unsupported {{.*}} range of parallel_for}}
  Kokkos::parallel_for("hello", teamPolicy, KOKKOS_LAMBDA(const Team &team) {
      ext();
  });
  // clang-format on
}

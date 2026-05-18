// Check that the correct warning is emitted when reducing a parallel_reduce
// and that we fall back to the default C++ lowering. Obviously when we support
// parallel_reduce, this test will have to be changed or removed.
//
// RUN: %kitxx -fkokkos -fkokkos-no-init -O1 --tapir=nolo -S -emit-llvm \
// RUN:     -fno-inline -o - %sysroot %s \
// RUN:     | FileCheck %s

#include <Kokkos_Core.hpp>

// CHECK-LABEL: double @f
// CHECK: call {{.+}}void @"_ZN6Kokkos15parallel_reduce
extern "C" double f(int n) {
  double result;

  // clang-format off
  // expected-warning-re@+1 {{kokkos - parallel_reduce is not supported{{.*}}}}
  Kokkos::parallel_reduce("sum", n, KOKKOS_LAMBDA(const int &i, double &lsum) {
    lsum += i;
  }, result);
  // clang-format on

  return result;
}

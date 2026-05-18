// Check that we correctly emit a warning and fall back to standard C++ lowering
// upon encountering a Kokkos::parallel_for with a functor instead of a lambda.
//
// RUN: %kitxx -fkokkos -fkokkos-no-init --tapir=nolo -O1 -S -emit-llvm \
// RUN:     -Xclang -verify %sysroot -o - %s \
// RUN:     | FileCheck %s

#include <Kokkos_Core.hpp>

extern "C" void ext(int);

struct Hello {
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const { ext(i); }
};

// CHECK-LABEL: void @f
// CHECK-SAME: i64{{.+}} %[[N:[^)]+]]
// CHECK: %[[HELLO:.+]] = alloca %struct.Hello
// CHECK: call void @_ZN6Kokkos12parallel_for{{.+}}(i64{{.+}} %[[N]], ptr{{.+}} %[[HELLO]])
extern "C" void f(size_t n) {
  // expected-warning@+1 {{kokkos: functors not supported in parallel_for}}
  Kokkos::parallel_for(n, Hello());
}

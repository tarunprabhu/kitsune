// Check that the tapir::target attribute on Kokkos::parallel_for statements is
// handled correctly.
//
// RUN: %kitxx -Xclang -verify -fsyntax-only -fkokkos -fkokkos-no-init  \
// RUN:   --tapir=serial -O1 %sysroot %s

#include <Kokkos_Core.hpp>

void f(float *A, int N) {
  // clang-format off

  // expected-error@+1 {{'tapir::target' attribute: unknown value}}
  [[tapir::target("i860")]]
  Kokkos::parallel_for(1024, KOKKOS_LAMBDA(const int i) {
    A[i] = i;
  });

  // expected-error@+1 {{'tapir::target' attribute requires a string}}
  [[tapir::target(serial)]]
  Kokkos::parallel_for(1024, KOKKOS_LAMBDA(const int i) {
    A[i] = i;
  });

  // expected-error@+1 {{'tapir::target' attribute takes one argument}}
  [[tapir::target()]]
  Kokkos::parallel_for(1024, KOKKOS_LAMBDA(const int i) {
    A[i] = i;
  });

  // expected-error@+1 {{'tapir::target' attribute takes one argument}}
  [[tapir::target("serial","-03")]]
  Kokkos::parallel_for(1024, KOKKOS_LAMBDA(const int i) {
    A[i] = i;
  });

  // expected-error@+1 {{'tapir::target' attribute: unsupported statement}}
  [[tapir::target("serial")]]
  if (N == 1) {
    Kokkos::parallel_for(1024, KOKKOS_LAMBDA(const int i) {
      A[i] = i;
    });
  }

  // clang-format on
}

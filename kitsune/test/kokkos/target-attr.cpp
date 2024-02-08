// RUN: %kitxx -Xclang -verify -fsyntax-only -fkokkos -fkokkos-no-init \
// RUN:   -ftapir=serial %sysroot %s

#include <Kokkos_Core.hpp>

void f(float* A, int N) {
  // clang-format off
  [[tapir::target("i860")]] // expected-error {{unknown tapir target}}
  Kokkos::parallel_for(1024, KOKKOS_LAMBDA(const int i) {
    A[i] = i;
  });

  [[tapir::target(serial)]] // expected-error {{'target' attribute requires a string}}
  Kokkos::parallel_for(1024, KOKKOS_LAMBDA(const int i) {
    A[i] = i;
  });

  [[tapir::target()]] // expected-error {{'target' attribute takes one argument}}
  Kokkos::parallel_for(1024, KOKKOS_LAMBDA(const int i) {
    A[i] = i;
  });

  [[tapir::target("serial","-03")]] // expected-error {{'target' attribute takes one argument}}
  Kokkos::parallel_for(1024, KOKKOS_LAMBDA(const int i) {
    A[i] = i;
  });

  [[tapir::target("serial")]] // expected-error {{tapir target attribute on unsupported statement}}
  if (N == 1) {
    Kokkos::parallel_for(1024, KOKKOS_LAMBDA(const int i) {
      A[i] = i;
    });
  }
  // clang-format on
}

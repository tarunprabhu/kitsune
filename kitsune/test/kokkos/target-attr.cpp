// RUN: %kitxx -Xclang -verify -fsyntax-only -fkokkos -fkokkos-no-init \
// RUN:   -ftapir=serial %s

#include "Kokkos_Core.hpp"
#include <kitsune.h>

using namespace kitsune;

int main(int argc, char *argv[]) {
  mobile_ptr<float> Am(1024);
  float* [[kitsune::mobile]] A = Am.get();

  Kokkos::initialize(argc, argv);
  {
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
    if (argc == 1) {
      Kokkos::parallel_for(1024, KOKKOS_LAMBDA(const int i) {
        A[i] = i;
      });
    }
    // clang-format on
  }
  Kokkos::finalize();

  return 0;
}

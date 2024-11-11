// REQUIRES: kitsune-kokkos
// XFAIL: *
// FIXME: This currently crashes with a "nested sync regions at end of function"
// error. Really should figure out why this is happening.

// The serial target is always built, so this is safe to run.
// RUN: %kitxx -fkokkos -fkokkos-no-init -O2 -fno-exceptions -ftapir=serial -S -emit-llvm -o - %s | FileCheck %s

// Very simple test of kokkos with two common forms of the
// parallel_for construct.  We should be able to transform
// all constructs from lambda into simple loops...
#include <cstdio>
#include <Kokkos_Core.hpp>

const unsigned int NTIMES = 10;

int main (int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    // clang-format off
    Kokkos::parallel_for(NTIMES, KOKKOS_LAMBDA(const int i) {
      printf("hello from %i\n", i);
    });

    printf("\n");

    Kokkos::parallel_for("hello1", NTIMES, KOKKOS_LAMBDA(const int i) {
      printf("hello from %i\n", i);
    });
    // clang-format on
  }
  Kokkos::finalize();

  return 0;
}

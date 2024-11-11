// REQUIRES: kitsune-kokkos
// XFAIL: *
// There is currently a bug in Kitsune which causes it to crash.
// KITSUNE FIXME: Fix this bug so the compiler does not crash.

// The serial target is always built, so this is safe to run.
// RUN: %kitxx -fkokkos -fkokkos-no-init -ftapir=serial -O2 -S -emit-llvm -o - %s | FileCheck %s

// Very simple test of kokkos that uses a functor.  In a nutshell,
// given the potential for different compilation units, kitsune does
// not support this construct and it should fall back to the standard
// C++ code gen paths.
#include <cstdio>
#include <Kokkos_Core.hpp>

const unsigned int NTIMES = 10;

struct Hello {
  KOKKOS_INLINE_FUNCTION
  void operator() (const int i) const {
    printf("hello from %i\n", i);
  }
};

int main (int argc, char* argv[]) {
  Kokkos::initialize (argc, argv);
  {
    Kokkos::parallel_for(NTIMES, Hello());
  }
  Kokkos::finalize ();

  return 0;
}

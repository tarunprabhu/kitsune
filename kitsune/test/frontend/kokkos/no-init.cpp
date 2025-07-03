// Check that the --kokkos-no-init option is handled correctly. This will
// remove calls to Kokkos::initialize and Kokkos::finalize.
//
// RUN: %kitxx --kokkos --tapir=none -S -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefix KOKKOS
//
// RUN: %kitxx --kokkos-no-init --tapir=none -S -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefix KOKKOS_NO_INIT
//
// KOKKOS: call void @_ZN6Kokkos10initializeERiPPc
// KOKKOS: call void @_ZN6Kokkos8finalizeEv
//
// KOKKOS_NO_INIT-NOT: call void @_ZN6Kokkos10initializeERiPPc
// KOKKOS_NO_INIT-NOT: call void @_ZN6Kokkos8finalizeEv

#include "Kokkos_Core.hpp"

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
  }
  Kokkos::finalize();
}

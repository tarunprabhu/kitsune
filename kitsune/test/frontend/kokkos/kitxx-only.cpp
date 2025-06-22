// The Kokkos-mode related options can only be used with the Kitsune frontend.
//
// RUN: not %clang --kokkos %s 2>&1 | FileCheck %s
// RUN: not %clang --kokkos-no-init %s 2>&1 | FileCheck %s
//
// CHECK: option '--kokkos{{.*}}' must be used with a Kitsune frontend

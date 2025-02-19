// The Kokkos-mode related options can only be used with the Kitsune frontend.
//
// RUN: not %clang -fkokkos %s 2>&1 | FileCheck %s
// RUN: not %clang -fkokkos-no-init %s 2>&1 | FileCheck %s

// CHECK: option '-fkokkos{{.*}}' must be used with a Kitsune frontend
